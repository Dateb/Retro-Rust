use std::collections::HashMap;

pub struct Controller {
    actions_to_buttons_bitmask_map: HashMap<usize, Vec<u8>>,
    num_buttons: usize,
    pub num_actions: usize,
}

impl Controller {
    pub fn new(button_combos: Vec<Vec<u64>>) -> Self {
        let max_button_value = button_combos.clone().iter()
            .flat_map(|inner| inner.iter())
            .copied()
            .max().expect("Max could not be extracted");

        let num_buttons = 64 - max_button_value.leading_zeros() as usize;

        let mut num_actions = 1;
        for combo in &button_combos {
            num_actions *= combo.len();
        }

        let actions_to_buttons_bitmask_map: HashMap<usize, Vec<u8>> =
            (0..num_actions)
                .map(|i| (i, Self::compute_button_bitmask(i, &button_combos, num_buttons)))
                .collect();

        Controller { actions_to_buttons_bitmask_map, num_buttons, num_actions }
    }

    fn compute_button_bitmask(
        mut action: usize,
        button_combos: &Vec<Vec<u64>>,
        num_buttons: usize
    ) -> Vec<u8> {
        let mut buttons_value: u64 = 0;
        for combo in button_combos {
            let current = action % combo.len();
            action /= combo.len();
            buttons_value |= combo[current];
        }
        let mut buttons_bitmask = vec![0u8; num_buttons];
        for i in 0..num_buttons {
            buttons_bitmask[i] = ((buttons_value >> i) & 1) as u8;
        }
        buttons_bitmask
    }

    pub fn get_button_bitmask(&self, action: usize) -> &Vec<u8> {
        self.actions_to_buttons_bitmask_map.get(&action).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_button_bitmask() {
        let mut button_combos = Vec::new();
        button_combos.push(vec![0, 16, 32]);
        button_combos.push(vec![0, 64, 128]);
        button_combos.push(vec![
            0,
            1,
            2,
            3,
            256,
            257,
            512,
            513,
            1024,
            1026,
            1536,
            2048,
            2304,
            2560,
        ]);
        let controller = Controller::new(button_combos);

        assert_eq!(controller.get_button_bitmask(0), vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(controller.get_button_bitmask(31), vec![1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0]);
        assert_eq!(controller.get_button_bitmask(82), vec![0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0]);
        assert_eq!(controller.get_button_bitmask(125), vec![0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1]);
    }
}
