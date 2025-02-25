# Copyright 2025 fat-tire and other contributors. Licensed under the GPL, version 3
import av
import folder_paths
import fractions
import numpy as np
import os
import torch


class MediaLoad:

    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = folder_paths.filter_files_content_types(os.listdir(input_dir), ["media", "image", "audio", "video"])
        return {"required": {
            "media_filename": (sorted(files), {"media_upload": True})}, # add /upload/media support to server.py?
            "optional": {
                "output_image_shape": (["B-H-W-C", "B-C-H-W"], {"default": "B-H-W-C"}),
                "batch_size": ("INT", {"default": 64, "min": 32, "max": 9999, "step": 1}),
                "frame_offset": ("INT", {"default": 0, "min": 0, "step": 1}),
                "frame_count": ("INT", {"default": 0, "min": 0, "step": 1}), }}

    RETURN_TYPES = ("IMAGE", "AUDIO", "FLOAT", "INT", "INT", "INT", "FLOAT", "INT")
    RETURN_NAMES = ("images", "audio", "fps", "width", "height", "num_frames", "duration", "audio_sample_rate")
    FUNCTION = "load_media"
    EXPERIMENTAL = True
    INPUT_NODE = True
    CATEGORY = "media"
    DESCRIPTION = ("Converts a mp4, mov, gif, avi, or other media file to images/audio. Also see the Save Media node,"
                   " which does the reverse-- saves image/audio to a media file. Lower batch_size if you get memory errors.")

    def load_media(self, media_filename, output_image_shape, batch_size, frame_offset, frame_count):
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
            batch_size = 16  # things slow down in my tests if it's bigger than this on cpu.
        container = av.open(os.path.join(folder_paths.get_input_directory(), media_filename))
        # Do video stream.
        start_time, target_time_in_seconds = 0, 0
        try:
            vid = container.streams.video[0]
            vid.thread_type = "AUTO"
            fps = vid.guessed_rate  # fps will be needed to calculate time for audio later
            num_frames = min(vid.frames - frame_offset, frame_count) if frame_count > 0 else vid.frames - frame_offset
            width = vid.codec_context.width
            height = vid.codec_context.height
            # pre-allocate final result tensor
            final_batched_tensor = torch.empty((num_frames, height, width, len(vid.codec_context.format.components)),
                                               dtype=torch.float32).cpu()
            start = 0
            # pre-allocate "fast" tensor to be re-used by gpu (if available) - size determined by user
            recycle_batched_tensor = torch.empty((batch_size, height, width, len(vid.codec_context.format.components)),
                                                 dtype=torch.float32).to(device)
            if len(vid.codec_context.format.components) == 4:  # if there's an alpha channel
                fmt = 'rgba'
            else:
                fmt = 'rgb24'  # save some memory
            container.seek(0, stream=vid)  # probably not needed, but just to know we're starting at the top.
            # for some reason, seeking to a specific frame is not reliable for video-- maybe because of key vs delta
            # frames, so I am just skipping past them to the frame_offset. Not the best way to do it, but seems to work.
            for _ in range(frame_offset):  # Skip the first 'frame_offset' frames for video (seek() works for audio)
                try:
                    next(container.decode(video=0))  # Decode, then throw away
                except StopIteration:  # we ran out of frames
                    break
            for i, frame in enumerate(container.decode(video=0)):
                target_time_in_seconds = i / fps if fps != 0 else 0.0 # probably could do this 1x when leaving loop
                # from https://pytorch.org/vision/0.8/transforms.html description of image tensor is:
                # (color, height, width) and batched is (B, C, H, W) and this is what comfyui apparently wants.
                tensor = (torch.from_numpy(frame.to_ndarray(format=fmt)).float() / 255.).to(device)
                if num_frames == 0:  # that is, the only frame in this file
                    final_batched_tensor = tensor.unsqueeze(0)
                    num_frames = 1
                    fps = 0
                else:
                    if i >= num_frames:
                        break
                    recycle_batched_tensor[i % batch_size] = tensor
                    if (i % batch_size == batch_size - 1) or i >= num_frames - 1:  # off-load frames from temp tensor
                        final_batched_tensor[start: min(start + batch_size, num_frames)] = recycle_batched_tensor[
                                                                                           :min(i % batch_size + 1,
                                                                                                batch_size)].cpu()
                        start += batch_size
                        if device == "cuda":
                            torch.cuda.empty_cache()
                        elif device == "mps":
                            torch.mps.empty_cache()
        except IndexError:  # no video stream available
            final_batched_tensor = torch.empty(0)
            width = 0
            height = 0
            num_frames = 0
            fps = 0
        # now audio
        try:
            audio_stream = container.streams.audio[0]
            audio_stream.thread_type = "AUTO"
            sample_rate = audio_stream.sample_rate
            audio_frames = []
            container.seek(0, stream=audio_stream)  # reset to beginning just in case.
            if frame_offset != 0 and fps != 0:
                start_time = int(round((frame_offset / fps) / audio_stream.time_base))
                container.seek(start_time, stream=audio_stream, backward=True)
            frame_duration_seconds = 0
            # have a resampler standing by in case this is 5.1 or something. Will mix to mono
            resampler = av.AudioResampler(
                format=audio_stream.format,  # Important: Use the original format
                rate=sample_rate,  # And the original sample rate
                layout="mono",
            )
            for i, frame in enumerate(container.decode(audio=0)):  # grab the audio samples
                if frame.layout.name not in ["mono", "stereo"]:
                    frame = resampler.resample(frame)[0]  # resample if too many channels
                audio_data = frame.to_ndarray().astype(np.float32)
                audio_frames.append(audio_data)
                frame_duration_seconds += fractions.Fraction(frame.samples, sample_rate)
                if frame_count != 0 and frame_duration_seconds >= target_time_in_seconds:
                    break
            if audio_frames:
                audio_data = np.concatenate(audio_frames, axis=1)
                audio_data = audio_data.reshape(-1,
                                                audio_data.shape[0] if audio_data.ndim == 1 else audio_data.shape[1])
                # Shave off extra samples if audio frames overrun video frames' duration due to a/v frame misalignment
                if target_time_in_seconds > 0:
                    if len(audio_data) > int(target_time_in_seconds * sample_rate):
                        audio_data = audio_data[:target_audio_samples]
                # normalize pcm audio data, or it sounds like complete crap.
                # See https://realpython.com/python-wav-files
                normalization_numbers = {  # what you divide by to normalize and the offset for unsigned #s.
                    's8': 128.0,
                    's16': 32768.0,
                    's32': 2147483648.0,
                    'u8': 128.0,
                    'u16': 32768.0,
                    'u32': 2147483648.0,
                }
                normalization_number = normalization_numbers.get(audio_stream.codec_context.format.name, 1.0)
                offset = normalization_number if audio_stream.codec_context.format.name in ['u8', 'u16', 'u32'] else 0
                audio_data = ((audio_data - offset) / normalization_number).astype(np.float32)
                waveform = torch.from_numpy(audio_data).to(device)  # look ma, don't need torchaudio.load!
            else:  # no frames in audio
                waveform = torch.empty(0)
                sample_rate = None
        except IndexError:  # no audio stream available at all
            waveform = torch.empty(0)
            sample_rate = None
        sound = {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}
        container.close()
        return (final_batched_tensor if output_image_shape == "B-H-W-C" else final_batched_tensor.permute(0, 3, 1, 2),
                sound, float(fps), int(width), int(height), int(num_frames), float(target_time_in_seconds),
                int(sample_rate or 0))
