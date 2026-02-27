#pragma once

#include <cstdint>
#include <cstddef>
#include <string>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle
typedef struct RetroEmulator RetroEmulator;
struct RetroGameData;

/////////////////////////////////////
///////////Create/Free///////////////
/////////////////////////////////////

RetroEmulator* emulator_new();
void emulator_free(RetroEmulator* handle);

/////////////////////////////////////
/////////////Methods/////////////////
/////////////////////////////////////

bool emulator_load_rom(RetroEmulator* h, const char* rom_path);
void emulator_run(RetroEmulator* handle);

bool emulator_set_state(RetroEmulator* h, const char* state_data, size_t state_size);

int emulator_get_screen_width(RetroEmulator* h);
int emulator_get_screen_height(RetroEmulator* h);
bool emulator_get_screen(RetroEmulator* h, int width, int height, uint8_t* out_rgb);
void emulator_set_button_mask(RetroEmulator* h, const uint8_t* mask, size_t num_buttons, unsigned player);
void emulator_set_key(RetroEmulator* h, int port, int key, bool active);
void emulator_configure_data(RetroEmulator* h, RetroGameData* data);
bool load_core_info(const char* json);

std::string emulator_get_core(RetroEmulator* handle);

#ifdef __cplusplus
}
#endif
