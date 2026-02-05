#include "rust_retro_emulator.h"
#include "rust_retro_gamedata.h"
#include "coreinfo.h"
#include "emulator.h"
#include "imageops.h"
#include <stdlib.h>
#include <string>

struct RetroEmulator {
    Retro::Emulator* emulator;
};

/////////////////////////////////////
///////////Create/Free///////////////
/////////////////////////////////////

RetroEmulator* emulator_new() {
    RetroEmulator* h = new RetroEmulator;
    h->emulator = new Retro::Emulator();
    return h;
}

void emulator_free(RetroEmulator* h) {
    if (!h) return;
    delete h->emulator;
    delete h;
}

/////////////////////////////////////
/////////////Methods/////////////////
/////////////////////////////////////

bool emulator_load_rom(RetroEmulator* h, const char* rom_path) {
    return h->emulator->loadRom(rom_path);
}

void emulator_run(RetroEmulator* h) {
    if (h && h->emulator) {
        h->emulator->run();
    }
}

bool emulator_set_state(RetroEmulator* h, const char* state_data, size_t state_size) {
    return h->emulator->unserialize(state_data, state_size);
}

int emulator_get_screen_width(RetroEmulator* h) {
    long width = h->emulator->getImageWidth();
    return static_cast<int>(width);
}

int emulator_get_screen_height(RetroEmulator* h) {
    long height = h->emulator->getImageWidth();
    return static_cast<int>(height);
}

bool emulator_get_screen(RetroEmulator* h, int width, int height, uint8_t* out_rgb) {
    Retro::Image in;
    if (h->emulator->getImageDepth() == 16) {
        in = Retro::Image(Retro::Image::Format::RGB565, h->emulator->getImageData(), width, height, h->emulator->getImagePitch());
    } else if (h->emulator->getImageDepth() == 32) {
        in = Retro::Image(Retro::Image::Format::RGBX888, h->emulator->getImageData(), width, height, h->emulator->getImagePitch());
    }
    Retro::Image out(Retro::Image::Format::RGB888, out_rgb, width, height, width * 3);

    in.copyTo(&out);
    return true;
}

void emulator_set_key(RetroEmulator* h, int port, int key, bool active) {
    h->emulator->setKey(port, key, active);
}

void emulator_set_button_mask(RetroEmulator* h, const uint8_t* mask, size_t num_buttons, unsigned player) {
    if (!h || !h->emulator) return;
    if (!mask) return;

    // Optional: bounds checks if you have constants
    // if (num_buttons > N_BUTTONS) return;
    // if (player >= MAX_PLAYERS) return;

    for (size_t key = 0; key < num_buttons; ++key) {
        h->emulator->setKey(player, (int)key, mask[key] != 0);
    }
}

bool load_core_info(const char* json) {
    return Retro::loadCoreInfo(json);
}

void emulator_configure_data(RetroEmulator* h, RetroGameData* data) {
    h->emulator->configureData(data->data);
}

std::string emulator_get_core(RetroEmulator* handle) {
    return handle->emulator->core();
}


