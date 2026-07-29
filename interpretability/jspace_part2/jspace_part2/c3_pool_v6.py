# Stage-3 bank, v6 increment — closing the last gap after the G5 audit.
#
# WHY v6 EXISTS, and what it corrects. v4 and v5 were authored as "new"
# families, but the G5 duplicate-fact audit found that nine of them
# re-expressed an EXISTING relation over an existing template — in four
# cases with literally the same fact on both sides ("The second-longest
# river in Africa is the ..." appears in v2 and v4). Those nine were
# merged into their originals in CANONICAL_FAMILY, which is why the count
# fell from 63 to 56 clusters with >=3 capable items. The audit did its
# job; the fix is more relations, not looser bookkeeping.
#
# v6 therefore authors relations that appear NOWHERE in the bank: not a
# work->creator, not an entity->category, not an ordinal superlative, not
# a structured count, not a process name, not a definition->term. Each
# family below is checked against RELATION_GROUP before scoring.
from __future__ import annotations

FAMILIES_V6: dict[str, list[tuple[str, str]]] = {
    "loanword_source_language": [
        ("The English word tsunami was borrowed from", "Japanese"),
        ("The English word safari was borrowed from", "Swahili"),
        ("The English word algebra was borrowed from", "Arabic"),
        ("The English word tycoon was borrowed from", "Japanese"),
        ("The English word bungalow was borrowed from", "Hindi"),
        ("The English word sauna was borrowed from", "Finnish"),
        ("The English word kindergarten was borrowed from", "German"),
        ("The English word ketchup ultimately came from", "Chinese"),
    ],
    "historical_period_successor": [
        ("The historical period that followed the Bronze Age is the", "Iron"),
        ("The historical period that followed the Middle Ages in Europe is the", "Renaissance"),
        ("The geological period that followed the Jurassic is the", "Cretaceous"),
        ("The geological period that followed the Triassic is the", "Jurassic"),
        ("The geological period that followed the Cambrian is the", "Ordovician"),
        ("The historical period that followed the Renaissance is the", "Enlightenment"),
        ("The geological epoch that followed the Pleistocene is the", "Holocene"),
        ("The geological period that followed the Devonian is the", "Carboniferous"),
    ],
    "abbreviation_expansion": [
        ("In computing, the abbreviation RAM stands for random access", "memory"),
        ("In computing, the abbreviation SQL stands for structured query", "language"),
        ("In computing, the abbreviation DNS stands for domain name", "system"),
        ("In medicine, the abbreviation MRI stands for magnetic resonance", "imaging"),
        ("In medicine, the abbreviation ECG stands for electrocardio", "gram"),
        ("In finance, the abbreviation IPO stands for initial public", "offering"),
        ("In physics, the abbreviation LASER stands for light amplification by stimulated emission of", "radiation"),
        ("In biology, the abbreviation ATP stands for adenosine tri", "phosphate"),
    ],
    "color_mixing_result": [
        ("Mixing the pigments blue and yellow produces", "green"),
        ("Mixing the pigments red and yellow produces", "orange"),
        ("Mixing the pigments red and blue produces", "purple"),
        ("Mixing the light colours red and green produces", "yellow"),
        ("Mixing the light colours green and blue produces", "cyan"),
        ("Mixing the light colours red and blue produces", "magenta"),
        ("Mixing black pigment with white pigment produces", "grey"),
        ("Mixing red pigment with white pigment produces", "pink"),
    ],
    "verb_for_animal_sound": [
        ("The verb for the sound a horse makes is to", "neigh"),
        ("The verb for the sound a donkey makes is to", "bray"),
        ("The verb for the sound a frog makes is to", "croak"),
        ("The verb for the sound a crow makes is to", "caw"),
        ("The verb for the sound a snake makes is to", "hiss"),
        ("The verb for the sound a wolf makes is to", "howl"),
        ("The verb for the sound a dove makes is to", "coo"),
        ("The verb for the sound a sheep makes is to", "bleat"),
    ],
    "tool_used_by_trade": [
        ("The tradesperson who works primarily with a trowel is a", "bricklayer"),
        ("The tradesperson who works primarily with a plane and chisel is a", "carpenter"),
        ("The tradesperson who works primarily with an anvil is a", "blacksmith"),
        ("The tradesperson who works primarily with a last is a", "cobbler"),
        ("The tradesperson who works primarily with a kiln is a", "potter"),
        ("The tradesperson who works primarily with a loom is a", "weaver"),
        ("The tradesperson who works primarily with a lathe is a", "turner"),
        ("The tradesperson who works primarily with a bellows and forge is a", "smith"),
    ],
    "blood_type_compatibility": [
        ("The blood type known as the universal donor is O", "negative"),
        ("The blood type known as the universal recipient is AB", "positive"),
        ("A person with type A blood carries antibodies against type", "B"),
        ("A person with type B blood carries antibodies against type", "A"),
        ("The blood group system based on the D antigen is the", "Rhesus"),
        ("A person with type O blood carries antigens numbering", "zero"),
        ("A person with type AB blood carries antibodies numbering", "zero"),
        ("Blood type incompatibility in pregnancy most often involves the", "Rhesus"),
    ],
    "crop_climate_region": [
        ("The climate zone in which coffee is chiefly grown is", "tropical"),
        ("The climate zone in which dates are chiefly grown is", "arid"),
        ("The climate zone in which olives are chiefly grown is", "Mediterranean"),
        ("The climate zone in which rice paddies are chiefly grown is", "tropical"),
        ("The climate zone in which barley grows farther north than wheat is", "temperate"),
        ("The climate zone in which cacao is chiefly grown is", "tropical"),
        ("The climate zone in which grapes for wine are chiefly grown is", "temperate"),
        ("The climate zone in which sugarcane is chiefly grown is", "tropical"),
    ],
    "material_property_use": [
        ("Tungsten is chosen for lamp filaments because of its high melting", "point"),
        ("Copper is chosen for electrical wiring because of its high", "conductivity"),
        ("Lead is chosen for radiation shielding because of its high", "density"),
        ("Kevlar is chosen for body armour because of its high tensile", "strength"),
        ("Aerogel is chosen for insulation because of its low thermal", "conductivity"),
        ("Titanium is chosen for implants because of its", "biocompatibility"),
        ("Graphite is chosen for pencil leads because of its", "softness"),
        ("Diamond is chosen for cutting tools because of its", "hardness"),
    ],
    "direction_of_change": [
        ("As altitude increases, atmospheric pressure", "decreases"),
        ("As a gas is compressed at constant temperature, its pressure", "increases"),
        ("As the wavelength of light increases, its frequency", "decreases"),
        ("As an object approaches the speed of light, its relativistic mass", "increases"),
        ("As temperature rises, the solubility of most solids in water", "increases"),
        ("As temperature rises, the solubility of gases in water", "decreases"),
        ("As a spring is stretched further, the restoring force", "increases"),
        ("As depth in the ocean increases, water temperature generally", "decreases"),
    ],
    "body_of_water_between": [
        ("The body of water separating Europe from Africa at its narrowest is the Strait of", "Gibraltar"),
        ("The body of water separating Asia from North America is the Bering", "Strait"),
        ("The body of water separating England from France is the English", "Channel"),
        ("The body of water separating Australia from Papua New Guinea is the Torres", "Strait"),
        ("The body of water separating Sri Lanka from India is the Palk", "Strait"),
        ("The body of water separating Sicily from mainland Italy is the Strait of", "Messina"),
        ("The body of water separating North and South Islands of New Zealand is Cook", "Strait"),
        ("The body of water separating Denmark from Sweden is the", "Oresund"),
    ],
    "orbital_or_rotation_period": [
        ("The planet with the longest day relative to its year is", "Venus"),
        ("The planet whose axis is tilted almost onto its orbital plane is", "Uranus"),
        ("The planet with the shortest year is", "Mercury"),
        ("The planet with the fastest rotation is", "Jupiter"),
        ("The planet whose rotation is retrograde is", "Venus"),
        ("The planet that takes about 84 Earth years to orbit the Sun is", "Uranus"),
        ("The planet that takes about 165 Earth years to orbit the Sun is", "Neptune"),
        ("The planet that takes about 29 Earth years to orbit the Sun is", "Saturn"),
    ],
}


def rows_v6() -> list[dict]:
    return [{"prompt": f"Fact: {p}", "answer": f" {a}", "family": fam,
             "pool": "v6"}
            for fam, items in FAMILIES_V6.items() for p, a in items]


if __name__ == "__main__":
    import sys
    from .c3_pool import RELATION_GROUP, canonical_family
    clash = [f for f in FAMILIES_V6
             if canonical_family(f) != f or f in RELATION_GROUP]
    r = rows_v6()
    print(f"v6: {len(FAMILIES_V6)} families, {len(r)} items; "
          f"pre-existing-relation clashes: {clash}")
    sys.exit(1 if clash else 0)
