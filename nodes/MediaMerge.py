# Copyright 2025 fat-tire and other contributors. Licensed under the GPL, version 3
import torch


class MediaMerge:

    @classmethod
    def INPUT_TYPES(cls):

        return {"required": {"foreground": ("IMAGE",),
                             "background": ("IMAGE",), },
                "optional": {
                    "blend": ("FLOAT", {"default": 1.0, "min": .01, "max": 1.0, "step": .01}),
                    "foreground_image_shape": (["B-H-W-C", "B-C-H-W"], {"default": "B-H-W-C"}),
                    "background_image_shape": (["B-H-W-C", "B-C-H-W"], {"default": "B-H-W-C"}),
                    "output_image_shape": (["B-H-W-C", "B-C-H-W"], {"default": "B-H-W-C"}),
                    "blend_mode": (["Normal", "Multiply", "Screen", "Overlay", "Soft Light", "Hard Light", "Difference",
                                    "Darken", "Lighten"],
                                   {"default": "Normal"}),
                    "swap_inputs": ("BOOLEAN", {"default": False, "label_on": "true", "label_off": "false"}), },
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"}, }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "merge_media"
    EXPERIMENTAL = True
    CATEGORY = "media"

    DESCRIPTION = ("Overlays a foreground over a background. Should handle alpha channels nicely. If the images are"
                   " different sizes, the foreground will be stretched to match the background. If the image counts"
                   " are different, the shorter layer (and its alpha channel) will be looped to fill the gap.")

    def pad_images(self, images, alpha, padding_size, device):
        # pad but also pad the alpha. Could be done in 2 passes but may as well combine
        frame_count = images.shape[0]
        repeat_factor = (padding_size + frame_count - 1) // frame_count
        # do this on CPU until I do some kind of batching in GPU memory, otherwise OOM.
        padding = images.cpu().repeat(repeat_factor, 1, 1, 1)[:padding_size]
        alpha_padding = alpha.cpu().repeat(repeat_factor, 1, 1, 1)[:padding_size]
        return (torch.cat([images.to(device), padding.to(device)], dim=0),
                torch.cat([alpha.to(device), alpha_padding.to(device)], dim=0))

    def merge_media(self, foreground, background, foreground_image_shape, background_image_shape,
                    output_image_shape, blend, blend_mode, swap_inputs=False, prompt=None, extra_pnginfo=None):
        device = "cpu"  # use cpu for now until batching is added or we'll probably blow up gpu memory
        # lets reshape if requested
        if foreground_image_shape == "B-C-H-W" and foreground is not None:
            foreground = foreground.permute(0, 2, 3, 1)
        if background_image_shape == "B-C-H-W" and background is not None:
            background = background.permute(0, 2, 3, 1)
        if swap_inputs:
            background, foreground = foreground, background
        # permutation needed to do a rescale as interpolate() requires (B C H W). This could be more elegant.
        foreground = torch.nn.functional.interpolate(
            foreground.permute(0, 3, 1, 2),
            size=background.shape[1:3],
            mode='bilinear',
            align_corners=False).permute(0, 2, 3, 1)  # Transpose back to B H W C
        if foreground.shape[3] == 4:  # 4 channels, alpha channel found!
            fg_alpha = foreground[..., 3].unsqueeze(-1)  # Add a channel dimension to alpha for later compatibility
        else:  # alpha is fully opaque. This can be optimized later to not have an extra alpha here.
            fg_alpha = torch.ones((foreground.shape[0], foreground.shape[1], foreground.shape[2], 1),
                                  dtype=torch.float32)
        if background.shape[3] == 4:  # 4 channels, alpha channel found!
            bg_alpha = background[..., 3].unsqueeze(-1)  # Add a channel dimension to alpha for later compatibility
        else:  # alpha is fully opaque. This can be optimized later to not have an extra alpha here.
            bg_alpha = torch.ones((background.shape[0], background.shape[1], background.shape[2], 1),
                                  dtype=torch.float32)
        # now that we have our A channel, we can chop out any alphas out of foreground/background (we'll add it later)
        if foreground.shape[-1] == 4:
            foreground = foreground[..., :3]
        if background.shape[-1] == 4:
            background = background[..., :3]
        # now make sure both batches have the same number of images by loop/repeating the shorter one
        # until they're the same size.
        foreground_batch = foreground.shape[0]  # this is the # of images
        background_batch = background.shape[0]
        if foreground_batch != background_batch:  # need to pad
            padding_size = abs(foreground_batch - background_batch)
            if background_batch > foreground_batch:
                foreground, fg_alpha = self.pad_images(foreground, fg_alpha, padding_size, device)
            else:
                background, bg_alpha = self.pad_images(background, bg_alpha, padding_size, device)
        # Mostly formulas from https://en.wikipedia.org/wiki/Blend_modes
        # Use mul_() (instead of "*") for in-place multiplication where there's no addition/subtraction, etc.
        # THESE FORMULAS ARE CONFUSING AF AND ARE PROBABLY TOTALLY WRONG. I think "Normal" is right tho :)
        # TODO: Factor out the repeated (foreground*alpha*blend) + ((1-alpha)*background*blend)
        # TODO: Add batching so the compositing/blending is done on the GPU. Right now it blows up if too many frames.
        match blend_mode:
            case "Multiply":
                # f(a,b) = ab
                composited_images = (foreground.to(device).mul_(background.to(device))).mul_(fg_alpha.to(device)).mul_(
                    blend) + background.to(device).mul_((1 - fg_alpha.to(device).mul_(blend)))
            case "Screen":
                # f(a,b) = 1 - (1-a)(1-b)
                composited_images = ((1 - (1 - foreground.to(device)) * (1 - background.to(device)))
                                     .mul_(fg_alpha.to(device)).mul_(blend) + background.to(device).mul_(
                    1 - fg_alpha.to(device).mul_(blend)))
            case "Overlay":
                # f(a,b) = 2ab (if a < .5) ELSE 1 - 2(1-a)(1-b)
                composited_images = torch.where(foreground.to(device) < 0.5, (2 * foreground.to(device).mul_(
                    background.to(device))).mul_(blend), 1 - 2 * (1 - foreground.to(device)) *
                                                (1 - background.to(device))).mul_(fg_alpha.to(device)).mul_(
                    blend) + background.to(device).mul_(1 - fg_alpha.to(device).mul_(blend))
            case "Hard Light":
                # f(a,b) = 2ab (if b < .5) ELSE 1 -2(1-a)(1-b)  # similar to above, switch condition
                composited_images = torch.where(background.to(device) < 0.5, (2 * foreground.to(device).mul_(
                    background.to(device))).mul_(blend), 1 - 2 * (1 - foreground.to(device)) *
                                                (1 - background.to(device))).mul_(fg_alpha.to(device)).mul_(
                    blend) + background.to(device).mul_(1 - fg_alpha.to(device).mul_(blend))
            case "Soft Light":  # use pegtop's formula from wikipedia, not photoshop as its supposedly better (?)
                # f(a,b) = (1-2b)a^2 + 2ba
                composited_images = (
                        (((1 - 2 * background.to(device)).mul_(
                            foreground.to(device).mul_(foreground.to(device))))  # part1
                         + (2 * (background.to(device).mul_(foreground.to(device)))))  # part2
                        .mul(fg_alpha.to(device)).mul_(blend)
                        + background.to(device).mul_((1 - fg_alpha.to(device).mul_(blend))))
            case "Difference":
                # f(a,b) = abs(a-b)
                composited_images = torch.abs(foreground.to(device) - background.to(device)).mul_(
                    fg_alpha.to(device)).mul_(
                    blend) + background.to(device).mul_((1 - fg_alpha.to(device).mul_(blend)))
            case "Darken":
                # f(a,b) = min(a,b)
                composited_images = ((torch.minimum(foreground.to(device), background.to(device)))
                                     .mul_(fg_alpha.to(device)).mul_(blend) + background.to(device).mul_(
                    (1 - fg_alpha.to(device).mul_(blend))))
            case "Lighten":
                # f(a,b) = max(a,b)
                composited_images = ((torch.maximum(foreground.to(device), background.to(device)))
                                     .mul_(fg_alpha.to(device)).mul_(blend) + background.to(device).mul_(
                    (1 - fg_alpha.to(device).mul_(blend))))
            case _:  # "Normal"
                composited_images = foreground.to(device).mul_(fg_alpha.to(device)).mul_(blend) + background.to(
                    device).mul_(1 - fg_alpha.to(device).mul(blend))
        # next do alpha blending. Assume this is the same for all blend modes, which maybe it's not (?)
        final_alpha = fg_alpha.cpu()
        final_alpha.add_(bg_alpha.cpu().mul_(1 - fg_alpha.cpu()))
        # and bring it all together
        final_result = torch.cat([composited_images.cpu(), final_alpha.cpu()], dim=3)
        return final_result.unsqueeze(0) if output_image_shape == "B-H-W-C" \
            else final_result.permute(0, 3, 1, 2).unsqueeze(0)
