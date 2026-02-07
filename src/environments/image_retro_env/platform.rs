use std::fmt;
use std::fmt::{Display, Formatter};

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

impl Display for Platform {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let s = match self {
            Platform::Atari => "Atari",
            Platform::GB => "GB",
            Platform::GBA => "GBA",
            Platform::NES => "NES",
            Platform::SNES => "SNES",
            Platform::Genesis => "Genesis",
            Platform::PCE => "PCE"
        };
        write!(f, "{}", s)
    }
}