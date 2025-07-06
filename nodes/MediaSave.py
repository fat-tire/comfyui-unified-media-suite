# Copyright 2025 fat-tire and other contributors. Licensed under the GPL, version 3
import av
import folder_paths
import fractions
import io
import json
import numpy as np
import os
import torch
import torchaudio
import torchvision
from comfy.cli_args import args


class MediaSave:

    @classmethod
    def INPUT_TYPES(cls):
        # need a temporary outputContainer to grab the codecs and pix_fmts for the UI
        with av.open(os.devnull, "w", "mp4") as cont:
            supported_codecs = sorted(cont.supported_codecs)
            vid_cods, aud_cods, vid_pix_fmts = [], [], []
            for candidate in supported_codecs:
                try:
                    c = av.codec.Codec(candidate, 'w')
                    if c.type == 'video' and c.video_formats is not None:
                        vid_cods.append(candidate)
                        vid_pix_fmts.extend([f.name for f in c.video_formats])
                    elif c.type == 'audio':
                        aud_cods.append(candidate)
                except av.codec.codec.UnknownCodecError:
                    pass
            vid_pix_fmts = list(set(vid_pix_fmts))  # unique
        cont.close()  # close the temporary container!
        presets = ["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "veryslow"]
        tunes = ["film", "animation", "grain", "stillimage", "fastdecode", "zerolatency", "none"]
        return {"required": {
            "filename_prefix": ("STRING", {"default": "MediaOut"}),
            "format": (sorted(["mp4", "mov", "webp", "gif", "apng", "avi", "flac", "wmv", "mkv", "png", "mpeg",
                               "mp3", "aac", "mjpeg"]), {"default": "mp4"}),  # Use "ffmpeg -formats" for the full list.
            "video_codec": (sorted(vid_cods), {"default": "libx264"}),
            "audio_codec": (sorted(aud_cods), {"default": "aac"}),
            "pix_fmt": (sorted(vid_pix_fmts), {"default": "yuv420p"}),
            "crf": ("INT", {"default": 23, "min": 1, "max": 51, "step": 1}),
            "preset": (presets, {"default": "medium"}), "tune": (tunes, {"default": "none"}), },
            "optional": {"fps": ("FLOAT", {"default": 24., "min": 1., "max": 1000., "step": 1.}),
                         "audio_sample_rate": ("INT", {"default": 48000, "step": 1000, "min": 0, "max": 192000}),
                         "images": ("IMAGE",), "audio": ("AUDIO", {"default": None}),
                         "width": ("INT", {"default": 0}), "height": ("INT", {"default": 0}),
                         "input_image_shape": (["B-H-W-C", "B-C-H-W"], {"default": "B-H-W-C"}),
                         "save_metadata": ("BOOLEAN", {"default": True})},
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"}, }

    RETURN_TYPES = ()
    FUNCTION = "save_to_media"
    OUTPUT_NODE = True
    EXPERIMENTAL = True
    CATEGORY = "media"

    DESCRIPTION = ("Saves images and/or audio to a file popular media formats. Also see the Load Media node, which does"
                   " the reverse-- loads a media file and provides images and audio.")

    def save_to_media(self, filename_prefix, format, video_codec, audio_codec, pix_fmt, crf, preset, tune,
                      width, height, input_image_shape, save_metadata, fps=24.0, audio_sample_rate=48000,
                      images=None, audio=None, prompt=None, extra_pnginfo=None):
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        if input_image_shape == "B-C-H-W" and images is not None:
            images = images.permute(0, 2, 3, 1)
        approved_sound_formats = ["mp4", "mov", "avi", "wmv", "mkv", "mp3", "aac",
                                  "flac"]  # Some, like gif, don't have sound.
        filename_prefix += ""
        results, pil_images, metadata = [], [], {}  # did you know you can define multiple items on one line?
        if images is not None and images.numel() == 0:
            images = None
        if images is None and audio is None:
            raise Exception("No audio or input detected. Nothing to do!")
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory(),
            0 if images is None else images[0].shape[1], 0 if images is None else images[0].shape[0])
        if not args.disable_metadata:
            if prompt is not None:
                metadata["prompt"] = json.dumps(prompt)
            if extra_pnginfo is not None:
                for x in extra_pnginfo:
                    metadata[x] = json.dumps(extra_pnginfo[x])
        alpha = False
        if images is not None:
            for image in images:  # props to @comfyanonymous in comfyui's eff24ea for showing how to do this.
                clipped = torch.clip(255. * image, 0, 255).to(torch.uint8).to(device).permute(2, 0, 1)
                img = torchvision.transforms.functional.to_pil_image(clipped)
                if images.shape[3] == 4:
                    alpha = True
                pil_images.append(img)
        file = f"{filename}_{counter:05}_.{format}"
        av.logging.set_level(av.logging.VERBOSE)
        output = av.open(os.path.join(full_output_folder, file), 'w')
        kind = "video"
        if format in ["gif", "webp", "png", "apng", "mjpeg"]:  # these technically aren't containers and have no codec
            video_codec = "png" if format in ["png", "png"] else format
            kind = "images"
        if format in ["mp3", "aac", "flac"]:
            audio_codec = format
            images = None
            kind = "audio"
        out_stream = None
        if images is not None:
            digits = 100000  # hack: FPS is of a fraction, but we're using a float, so convert to a "fake" fraction.
            out_stream = output.add_stream(video_codec, framerate=fractions.Fraction(numerator=int(fps * digits),
                                                                                     denominator=digits))
            out_stream.codec_context.options["crf"] = str(crf)  # See https://trac.ffmpeg.org/wiki/Encode/H.264
            if format not in ['webp']:  # chokes for some reason
                out_stream.codec_context.options["preset"] = preset
            if fps != 0:  # divide by zero issue
                out_stream.codec_context.time_base = fractions.Fraction(digits, int(fps * digits))
                out_stream.time_base = fractions.Fraction(digits, int(fps * digits))
            if tune != "none":
                out_stream.codec_context.options["tune"] = tune
            out_stream.thread_type = "AUTO"
            out_stream.width = pil_images[0].width if width == 0 else width
            out_stream.height = pil_images[0].height if height == 0 else height
        audio_output_stream = None
        if audio and format in approved_sound_formats:
            audio["sample_rate"] = audio_sample_rate
            audio_output_stream = output.add_stream(audio_codec, audio["sample_rate"])
            audio_output_stream.codec_context.time_base = fractions.Fraction(1, audio["sample_rate"]) \
                if audio["sample_rate"] != 0 else 0
            audio_output_stream.time_base = fractions.Fraction(1, audio["sample_rate"])
        if metadata is not None and not args.disable_metadata and save_metadata:
            # unsure where metadata goes. use "comment" if avail. https://wiki.multimedia.cx/index.php/FFmpeg_Metadata
            if format in ["flv", "flac", "png"]:
                output.metadata["Vendor"] = "ComfyUI"
                for key, value in metadata.items():
                    output.metadata[key] = value
            elif format in ["mp4", 'asf', 'wmv', 'wma', "avi", 'mov', "mkv"]:
                output.metadata["comment"] = json.dumps(metadata)
            else:  # mp3, etc.
                output.metadata['encoded_by'] = json.dumps(metadata)
        # Find valid pix_fmt for each codec with ffmpeg -h encoder=codec_name - transparency uses alpha pix_fmt
        match format:
            case "gif":
                pix_fmt = "rgb8"
            case "webp":
                if alpha or pix_fmt not in [pf.name for pf in av.codec.Codec("webp", 'w').video_formats]:
                    pix_fmt = "yuva420p"
            case "apng" | "png":
                if alpha or pix_fmt not in [pf.name for pf in av.codec.Codec("apng", 'w').video_formats]:
                    pix_fmt = "rgba"
            case "av1":
                if alpha or pix_fmt not in [pf.name for pf in av.codec.Codec("av1", 'w').video_formats]:
                    pix_fmt = "gray12le"
            case "prores_ks" | "prores":
                if alpha or pix_fmt not in [pf.name for pf in av.codec.Codec("prores_ks", 'w').video_formats]:
                    pix_fmt = "yuva444p10le"
            case "ffv1":
                if alpha or pix_fmt not in [pf.name for pf in av.codec.Codec("ffv1", 'w').video_formats]:
                    pix_fmt = "yuva420p"
            case "mjpeg":
                if alpha or pix_fmt not in [pf.name for pf in av.codec.Codec("mjpeg", 'w').video_formats]:
                    pix_fmt = "yuvj422p"
            case "mp3" | "flac":
                pix_fmt = None
        if pix_fmt is not None and out_stream is not None:
            out_stream.pix_fmt = pix_fmt
        image_cnt = len(pil_images) if pil_images is not None else 0
        if images is not None:
            for i in range(0, image_cnt):  # mux the images
                if alpha:  # only from_ndarray supports rgba
                    outFrame = av.VideoFrame.from_ndarray(np.array(pil_images[i].convert("RGBA")), format="rgba")
                else:
                    outFrame = av.VideoFrame.from_image(pil_images[i])
                out_packet = out_stream.encode(outFrame)
                output.mux(out_packet)
        if audio and (format in approved_sound_formats):
            # Save to a buffer instead of a file, then mux in the audio from an output_stream based on the buffer.
            for (batch_number, waveform) in enumerate(audio["waveform"].cpu()):
                buff = io.BytesIO()
                torchaudio.save(buff, waveform, audio["sample_rate"], format="FLAC")  # write audio to buffer
                buff.seek(0)  # reset buffer for writing. This took a while to figure out
                audio_input_container = av.open(buff, "r")  # now read the buffer
                audio_input_stream = audio_input_container.streams.get(audio=0)[0]  # then get the audio stream
                audio_input_stream.thread_type = "AUTO"
                audio_output_stream.sample_rate = audio["sample_rate"]
                for frame in audio_input_container.decode(audio_input_stream):  # now mux the audio stream into file.
                    for i, packet in enumerate(audio_output_stream.encode(frame)):
                        packet.pts = frame.pts + i
                        packet.dts = frame.dts + i
                        output.mux(packet)
        if images is not None:  # flush the container and close
            out_packet = out_stream.encode(None)
            if out_packet is not None:
                output.mux(out_packet)
        output.close()  # done!
        results.append({"filename": file, "subfolder": subfolder, "type": "output", })
        return {"ui": {kind: results, "animated": (image_cnt != 1,)}}
