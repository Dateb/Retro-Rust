use image::{imageops::resize, imageops::FilterType, ImageBuffer, Luma, Rgb, RgbImage};

#[derive(Debug)]
pub struct FrameProcessor {
    resized_width: u32,
    resized_height: u32
}

impl FrameProcessor {
    pub fn new(resized_width: u32, resized_height: u32) -> Self {
        Self {
            resized_width,
            resized_height,
        }
    }

    pub fn process_frame(&self, buffer: Vec<u8>, w: u32, h: u32) -> Result<Vec<f32>, String> {
        let frame: RgbImage = ImageBuffer::from_raw(w, h, buffer)
            .ok_or("Buffer size does not match width × height × 3")?;

        let resized_frame = self.resize_frame(&frame);
        let gray_frame = self.to_grayscale_frame(resized_frame);
        Ok(self.flatten_frame(gray_frame))
    }

    fn resize_frame(&self, frame: &RgbImage) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
        resize(
            frame,
            self.resized_width,
            self.resized_height,
            FilterType::Nearest
        )
    }

    fn to_grayscale_frame(&self, frame: ImageBuffer<Rgb<u8>, Vec<u8>>)
        -> ImageBuffer<Luma<f32>, Vec<f32>> {
        ImageBuffer::from_fn(frame.width(), frame.height(), |x, y| {
            let pixel = frame.get_pixel(x, y);
            // Standard grayscale: 0.299 R + 0.587 G + 0.114 B
            let gray_val = (0.299 * pixel[0] as f32
                + 0.587 * pixel[1] as f32
                + 0.114 * pixel[2] as f32) / 255.0;
            Luma([gray_val])
        })
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
        let frame_processor = FrameProcessor::new(resized_width, resized_height);

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
        let frame_processor = FrameProcessor::new(resized_width, resized_height);

        let buffer_input = create_buffer_input();

        let processed_frame = frame_processor
            .process_frame(buffer_input.0, buffer_input.1, buffer_input.2)
            .expect("process_frame should succeed for valid buffer");

        assert_eq!(processed_frame[0], 0.299);
        assert_eq!(processed_frame[1], 0.587);
        assert_eq!(processed_frame[2], 0.114);
        assert_eq!(processed_frame[3], (0.299 * 255.0 + 0.587 * 255.0) / 255.0);
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