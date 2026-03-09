use std::time::Instant;
use fast_image_resize::{IntoImageView, PixelType, ResizeAlg, ResizeOptions, Resizer};
use fast_image_resize::images::Image;
use image::{imageops::resize, imageops::FilterType, ColorType, DynamicImage, ImageBuffer, ImageReader, Luma, Rgb, RgbImage};

#[derive(Debug)]
pub struct FrameProcessor<'a> {
    resized_width: u32,
    resized_height: u32,
    resizer: Resizer,
    resized_image: Image<'a>
}

impl<'a> FrameProcessor<'a> {
    pub fn new(resized_width: u32, resized_height: u32) -> Self {
        let resizer = Resizer::new();
        let resized_image = Image::new(
            resized_width,
            resized_height,
            PixelType::U8x3
        );
        Self {
            resized_width,
            resized_height,
            resizer,
            resized_image
        }
    }

    pub fn process_frame(&mut self, buffer: Vec<u8>, w: u32, h: u32) -> Result<Vec<f32>, String> {
        let frame: RgbImage = ImageBuffer::from_raw(w, h, buffer)
            .ok_or("Buffer size does not match width × height × 3")?;
        let resize_option = ResizeOptions {
            algorithm: ResizeAlg::Nearest,
            cropping: Default::default(),
            mul_div_alpha: false
        };

        self.resizer.resize(&frame, &mut self.resized_image, &resize_option).unwrap();
        
        let gray = self.rgb_to_gray_f32(
            self.resized_image.buffer(),
            self.resized_width,
            self.resized_height,
        );

        Ok(gray)
    }

    fn rgb_to_gray_f32(
        &self,
        rgb: &[u8],
        width: u32,
        height: u32,
    ) -> Vec<f32> {
        let mut out = Vec::with_capacity((width * height) as usize);

        for chunk in rgb.chunks_exact(3) {
            let r = chunk[0] as f32;
            let g = chunk[1] as f32;
            let b = chunk[2] as f32;

            // Same formula as before
            let gray = (0.299 * r + 0.587 * g + 0.114 * b) * (1.0 / 255.0);
            out.push(gray);
        }

        out
    }


    fn flatten_frame(&self, gray_image: ImageBuffer<Luma<f32>, Vec<f32>>) -> Vec<f32> {
        gray_image.pixels().map(|p| p[0]).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn processed_frame_correct_size() {
        let resized_width = 22;
        let resized_height = 8;
        let mut frame_processor = FrameProcessor::new(resized_width, resized_height);

        let buffer_input = create_buffer_input();

        let processed_frame = frame_processor
            .process_frame(buffer_input.0, buffer_input.1, buffer_input.2)
            .expect("process_frame should succeed for valid buffer");

        assert_eq!(
            processed_frame.len(),
            (resized_width * resized_height) as usize
        );
    }

    #[test]
    fn processed_frame_correct_pixel_values() {
        let resized_width = 2;
        let resized_height = 2;
        let mut frame_processor = FrameProcessor::new(resized_width, resized_height);

        let buffer_input = create_buffer_input();

        let processed_frame = frame_processor
            .process_frame(buffer_input.0, buffer_input.1, buffer_input.2)
            .expect("process_frame should succeed for valid buffer");

        assert!((processed_frame[0] - 0.299).abs() < 1e-6);
        assert!((processed_frame[1] - 0.587).abs() < 1e-6);
        assert!((processed_frame[2] - 0.114).abs() < 1e-6);
        assert!((processed_frame[3] - (0.299 * 255.0 + 0.587 * 255.0) / 255.0).abs() < 1e-6);
    }

    fn create_buffer_input() -> (Vec<u8>, u32, u32) {
        let frame_buffer = vec![
            255, 0, 0,
            0, 255, 0,
            0, 0, 255,
            255, 255, 0,
        ];

        (frame_buffer, 2, 2)
    }
}