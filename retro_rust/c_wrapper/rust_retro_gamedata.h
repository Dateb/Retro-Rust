#pragma once

#include <cstdint>
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle
namespace Retro {
    class GameData;
    class Scenario;
}
struct RetroGameData {
    Retro::GameData* data;
    Retro::Scenario* scenario;
};
typedef struct {
    int key;
    int* values;
    size_t num_values;
} RetroActionSet;

/////////////////////////////////////
///////////Create/Free///////////////
/////////////////////////////////////

RetroGameData* gamedata_new();
RetroActionSet* gamedata_valid_actions(RetroGameData* h, size_t* num_entries);
void gamedata_free_valid_actions(RetroActionSet* actions, size_t num_entries);

/////////////////////////////////////
/////////////Methods/////////////////
/////////////////////////////////////

bool gamedata_load(RetroGameData* h, const char* data_path, const char* scenario_path);
void gamedata_reset(RetroGameData* h);
void gamedata_update_ram(RetroGameData* h);
float gamedata_current_reward(RetroGameData* h);
float gamedata_total_reward(RetroGameData* h);
bool gamedata_is_done(RetroGameData* h);

#ifdef __cplusplus
}
#endif
