pub enum Platform {
    Atari,
    GB,
    GBA,
    NES,
    SNES,
    Genesis,
    PCE
}

impl Platform {
    pub fn as_str(&self) -> &'static str {
        match self {
            Platform::Atari => "Atari",
            Platform::GB => "GB",
            Platform::GBA => "GBA",
            Platform::NES => "NES",
            Platform::SNES => "SNES",
            Platform::Genesis => "Genesis",
            Platform::PCE => "PCE",
        }
    }

    pub fn rom_name(&self) -> &'static str {
        match self {
            Platform::Atari => "rom.a26",
            Platform::GB => "rom.gb",
            Platform::GBA => "rom.gba",
            Platform::NES => "rom.nes",
            Platform::SNES => "rom.sfc",
            Platform::Genesis => "rom.md",
            Platform::PCE => "rom.pce"
        }
    }
}
