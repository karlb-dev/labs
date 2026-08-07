# Stage-3 hard one-hop candidate pool (C3 expansion).
#
# WHY: the v1 pool (113 curated facts) was too easy — 68/113 sat at
# ceiling on Olmo-3-32B-Think, leaving only n=41 after the difficulty
# filter (evidence `c3-hard-onehop-dev-v1`, dev tier, frozen forever).
# The confirmatory design needs ~n=90+ hard items across >=30 INDEPENDENT
# families, and the addendum's clustering rule (§12.2) makes the family
# the true generation unit — so the pool is authored family-first: each
# family is one relation template, and items within a family share it.
# Analysis clusters on `family`; nothing here may be treated as 30
# independent draws of a single template (the pseudo-replication debt the
# addendum found in the SQL battery).
#
# DIFFICULTY TARGET: baseline answer lp in [-9, -1] on the anchor model
# (the v1 match rule). Facts are chosen to be genuinely known-but-
# unrehearsed: second-order superlatives, less-canonical exemplars of a
# canonical relation, and mid-frequency proper nouns. NO item's answer is
# a top-frequency completion of its prompt (the v1 failure mode).
#
# This module is DATA ONLY (no scoring, no partitioning). Partition into
# confirmatory/replication happens at preregistration freeze, user-gated,
# hashed before outcomes are viewed (prereg §Item pools).
from __future__ import annotations

# family -> (prompt template note, [(prompt, answer), ...])
# Prompts carry the battery's "Fact: " prefix (v1 convention) so the pool
# is scorable with the same instrument.
FAMILIES: dict[str, list[tuple[str, str]]] = {
    "chem_symbol_rare": [
        ("The chemical symbol for praseodymium is", "Pr"),
        ("The chemical symbol for ytterbium is", "Yb"),
        ("The chemical symbol for niobium is", "Nb"),
        ("The chemical symbol for rhenium is", "Re"),
        ("The chemical symbol for hafnium is", "Hf"),
        ("The chemical symbol for gadolinium is", "Gd"),
    ],
    "element_atomic_number_mid": [
        ("The element with atomic number 41 is", "niobium"),
        ("The element with atomic number 45 is", "rhodium"),
        ("The element with atomic number 52 is", "tellurium"),
        ("The element with atomic number 63 is", "europium"),
        ("The element with atomic number 77 is", "iridium"),
    ],
    "si_unit_derived": [
        ("The SI unit of magnetic flux is the", "weber"),
        ("The SI unit of inductance is the", "henry"),
        ("The SI unit of absorbed radiation dose is the", "gray"),
        ("The SI unit of equivalent radiation dose is the", "sievert"),
        ("The SI unit of catalytic activity is the", "katal"),
        ("The SI unit of luminous flux is the", "lumen"),
    ],
    "capital_less_cited": [
        ("The capital of Kyrgyzstan is", "Bishkek"),
        ("The capital of Bhutan is", "Thimphu"),
        ("The capital of Suriname is", "Paramaribo"),
        ("The capital of Eritrea is", "Asmara"),
        ("The capital of Turkmenistan is", "Ashgabat"),
        ("The capital of Malawi is", "Lilongwe"),
    ],
    "capital_former": [
        ("The former capital of Brazil, before Brasilia, was", "Rio"),
        ("The former capital of Nigeria, before Abuja, was", "Lagos"),
        ("The former capital of Kazakhstan, before Astana, was", "Almaty"),
        ("The former capital of Myanmar, before Naypyidaw, was", "Yangon"),
        ("The former capital of Tanzania, before Dodoma, was", "Dar"),
    ],
    "currency_less_cited": [
        ("The currency of Kyrgyzstan is the", "som"),
        ("The currency of Laos is the", "kip"),
        ("The currency of Myanmar is the", "kyat"),
        ("The currency of Angola is the", "kwanza"),
        ("The currency of Botswana is the", "pula"),
        ("The currency of Paraguay is the", "guarani"),
        ("The currency of Georgia is the", "lari"),
    ],
    "river_second_order": [
        ("The second-longest river in Africa is the", "Congo"),
        ("The second-longest river in South America is the", "Parana"),
        ("The second-longest river in Europe is the", "Danube"),
        ("The second-longest river in Asia is the", "Yellow"),
        ("The longest river entirely within Wales is the", "Towy"),
    ],
    "mountain_second_order": [
        ("The second-highest mountain in the world is", "K2"),
        ("The second-highest mountain in Africa is Mount", "Kenya"),
        ("The highest mountain in Turkey is Mount", "Ararat"),
        ("The highest mountain in Iran is Mount", "Damavand"),
        ("The highest mountain in the Alps outside France is Monte", "Rosa"),
    ],
    "lake_specific": [
        ("The largest lake in South America is Lake", "Titicaca"),
        ("The largest lake entirely within Canada is Great", "Bear"),
        ("The saltiest large lake in Africa is Lake", "Assal"),
        ("The largest lake in Central Asia after the Caspian is the",
         "Aral"),
    ],
    "moons_planetary": [
        ("The largest moon of Saturn is", "Titan"),
        ("The largest moon of Neptune is", "Triton"),
        ("The largest moon of Uranus is", "Titania"),
        ("The innermost of Jupiter's Galilean moons is", "Io"),
        ("The moon of Saturn with geysers at its south pole is",
         "Enceladus"),
    ],
    "star_brightest_constellation": [
        ("The brightest star in the constellation Lyra is", "Vega"),
        ("The brightest star in the constellation Cygnus is", "Deneb"),
        ("The brightest star in the constellation Aquila is", "Altair"),
        ("The brightest star in the constellation Auriga is", "Capella"),
        ("The brightest star in the constellation Bootes is", "Arcturus"),
    ],
    "author_less_canonical": [
        ("The author of the novel Buddenbrooks is Thomas", "Mann"),
        ("The author of the novel Kokoro is Natsume", "Soseki"),
        ("The author of the novel Independent People is Halldor",
         "Laxness"),
        ("The author of the novel Season of Migration to the North is "
         "Tayeb", "Salih"),
        ("The author of the novel The Leopard is Giuseppe Tomasi di",
         "Lampedusa"),
        ("The author of the novel Petals of Blood is Ngugi wa", "Thiong"),
    ],
    "playwright_work": [
        ("The playwright of the drama Hedda Gabler is Henrik", "Ibsen"),
        ("The playwright of the drama The Cherry Orchard is Anton",
         "Chekhov"),
        ("The playwright of the drama Six Characters in Search of an "
         "Author is Luigi", "Pirandello"),
        ("The playwright of the drama Blood Wedding is Federico Garcia",
         "Lorca"),
    ],
    "philosopher_work": [
        ("The philosopher who wrote Being and Time is Martin",
         "Heidegger"),
        ("The philosopher who wrote The Phenomenology of Spirit is Georg "
         "Wilhelm Friedrich", "Hegel"),
        ("The philosopher who wrote Beyond Good and Evil is Friedrich",
         "Nietzsche"),
        ("The philosopher who wrote A Theory of Justice is John", "Rawls"),
        ("The philosopher who wrote The Structure of Scientific "
         "Revolutions is Thomas", "Kuhn"),
    ],
    "composer_work_less_cited": [
        ("The composer of the opera Boris Godunov is Modest",
         "Mussorgsky"),
        ("The composer of the ballet The Rite of Spring is Igor",
         "Stravinsky"),
        ("The composer of the opera Peter Grimes is Benjamin", "Britten"),
        ("The composer of the tone poem Finlandia is Jean", "Sibelius"),
        ("The composer of the opera Turandot is Giacomo", "Puccini"),
    ],
    "painter_work_less_cited": [
        ("The painter of the work The Garden of Earthly Delights is "
         "Hieronymus", "Bosch"),
        ("The painter of the work Las Meninas is Diego", "Velazquez"),
        ("The painter of the work The Great Wave off Kanagawa is",
         "Hokusai"),
        ("The painter of the work Christina's World is Andrew", "Wyeth"),
        ("The painter of the work Nighthawks is Edward", "Hopper"),
    ],
    "architect_building": [
        ("The architect of the Sagrada Familia is Antoni", "Gaudi"),
        ("The architect of Fallingwater is Frank Lloyd", "Wright"),
        ("The architect of the Sydney Opera House is Jorn", "Utzon"),
        ("The architect of the Guggenheim Museum Bilbao is Frank",
         "Gehry"),
        ("The architect of the Villa Savoye is Le", "Corbusier"),
    ],
    "ancient_city_country": [
        ("The ancient city of Petra is in the modern country of",
         "Jordan"),
        ("The ancient city of Palmyra is in the modern country of",
         "Syria"),
        ("The ancient city of Persepolis is in the modern country of",
         "Iran"),
        ("The ancient city of Carthage is in the modern country of",
         "Tunisia"),
        ("The ancient city of Angkor is in the modern country of",
         "Cambodia"),
    ],
    "treaty_place": [
        ("The treaty that ended the Thirty Years War was the Peace of",
         "Westphalia"),
        ("The treaty that ended the Russo-Japanese War was the Treaty of",
         "Portsmouth"),
        ("The treaty that divided the New World between Spain and "
         "Portugal was the Treaty of", "Tordesillas"),
        ("The treaty that ended the War of the Spanish Succession was the "
         "Treaty of", "Utrecht"),
    ],
    "dynasty_china": [
        ("The Chinese dynasty that produced the Terracotta Army was the",
         "Qin"),
        ("The Chinese dynasty during which Marco Polo is said to have "
         "visited was the", "Yuan"),
        ("The Chinese dynasty that invented woodblock printing was the",
         "Tang"),
        ("The last imperial dynasty of China was the", "Qing"),
    ],
    "mythology_less_cited": [
        ("The Norse god who guards the Bifrost bridge is", "Heimdall"),
        ("The Greek goddess of the dawn is", "Eos"),
        ("The Greek titan who stole fire for humanity is", "Prometheus"),
        ("The Egyptian goddess of magic and healing is", "Isis"),
        ("The Hindu god with an elephant head is", "Ganesha"),
        ("The Mesopotamian hero of the oldest surviving epic is",
         "Gilgamesh"),
    ],
    "explorer_first": [
        ("The explorer who first reached the South Pole was Roald",
         "Amundsen"),
        ("The explorer who first circumnavigated the globe departed under "
         "Ferdinand", "Magellan"),
        ("The explorer who mapped much of the Australian coast was "
         "Matthew", "Flinders"),
        ("The explorer who crossed Antarctica's Weddell Sea ice with the "
         "Endurance was Ernest", "Shackleton"),
    ],
    "spacecraft_mission": [
        ("The spacecraft that orbited Saturn for thirteen years was",
         "Cassini"),
        ("The spacecraft that landed a probe on a comet was", "Rosetta"),
        ("The spacecraft that first flew past Pluto was New", "Horizons"),
        ("The rover that confirmed ancient water on Mars in 2004 was",
         "Opportunity"),
    ],
    "anatomy_specific": [
        ("The longest nerve in the human body is the", "sciatic"),
        ("The smallest bone in the human body is the", "stapes"),
        ("The largest artery in the human body is the", "aorta"),
        ("The bone that forms the human cheek is the", "zygomatic"),
        ("The membrane surrounding the human heart is the",
         "pericardium"),
    ],
    "enzyme_process": [
        ("The enzyme that unwinds DNA during replication is", "helicase"),
        ("The enzyme that joins DNA fragments is DNA", "ligase"),
        ("The enzyme in saliva that begins starch digestion is",
         "amylase"),
        ("The enzyme that fixes carbon in photosynthesis is",
         "rubisco"),
    ],
    "medical_discoverer": [
        ("The physician who introduced antiseptic surgery was Joseph",
         "Lister"),
        ("The physician who first described the circulation of blood was "
         "William", "Harvey"),
        ("The scientist who discovered the structure of insulin was "
         "Frederick", "Sanger"),
        ("The scientist whose X-ray images enabled the DNA double helix "
         "was Rosalind", "Franklin"),
    ],
    "physicist_named_effect": [
        ("The physicist for whom the effect of light scattering by "
         "molecules is named is Chandrasekhara", "Raman"),
        ("The physicist for whom the exclusion principle is named is "
         "Wolfgang", "Pauli"),
        ("The physicist for whom the equation of quantum wavefunctions is "
         "named is Erwin", "Schrodinger"),
        ("The physicist for whom the uncertainty-free complementarity "
         "principle of quantum mechanics is named is Niels", "Bohr"),
        ("The physicist for whom the constant in blackbody radiation is "
         "named is Max", "Planck"),
    ],
    "mathematician_theorem": [
        ("The mathematician for whom the last theorem about integer "
         "powers is named is Pierre de", "Fermat"),
        ("The mathematician who proved the incompleteness theorems is "
         "Kurt", "Godel"),
        ("The mathematician for whom a famous unsolved hypothesis about "
         "the zeta function is named is Bernhard", "Riemann"),
        ("The mathematician who founded modern topology's fundamental "
         "group is Henri", "Poincare"),
        ("The mathematician for whom the sieve for finding primes is "
         "named is", "Eratosthenes"),
    ],
    "economics_concept_person": [
        ("The economist for whom the curve relating tax rates to revenue "
         "is named is Arthur", "Laffer"),
        ("The statistician for whom the coefficient of inequality is "
         "named is Corrado", "Gini"),
        ("The mathematician for whom the equilibrium in game theory is "
         "named is John", "Nash"),
        ("The economist who wrote The General Theory of Employment, "
         "Interest and Money is John Maynard", "Keynes"),
    ],
    "programming_language_creator": [
        ("The programming language created by Guido van Rossum is",
         "Python"),
        ("The programming language created by Bjarne Stroustrup is",
         "C"),
        ("The programming language created by James Gosling is", "Java"),
        ("The programming language created by Yukihiro Matsumoto is",
         "Ruby"),
        ("The programming language created by John McCarthy is", "Lisp"),
    ],
    "language_family": [
        ("The language family to which Finnish belongs is", "Uralic"),
        ("The language family to which Swahili belongs is", "Bantu"),
        ("The language family to which Tamil belongs is", "Dravidian"),
        ("The language family to which Hungarian belongs is", "Uralic"),
        ("The language family to which Hebrew belongs is", "Semitic"),
    ],
    "script_writing_system": [
        ("The writing system used for Japanese particles and inflections "
         "is", "hiragana"),
        ("The writing system used for the Hindi language is",
         "Devanagari"),
        ("The writing system invented for Korean in the fifteenth century "
         "is", "Hangul"),
        ("The script used to write Ethiopian Amharic is", "Ge"),
    ],
    "periodic_group_name": [
        ("The name of the periodic table group containing fluorine and "
         "chlorine is the", "halogens"),
        ("The name of the periodic table group containing lithium and "
         "sodium is the alkali", "metals"),
        ("The name of the periodic table group containing calcium and "
         "magnesium is the alkaline earth", "metals"),
        ("The name of the periodic table series containing uranium is the",
         "actinides"),
    ],
    "mineral_property": [
        ("The mineral defining hardness 10 on the Mohs scale is",
         "diamond"),
        ("The mineral defining hardness 7 on the Mohs scale is", "quartz"),
        ("The main ore mineral of aluminium is", "bauxite"),
        ("The main ore mineral of mercury is", "cinnabar"),
        ("The mineral known as fool's gold is", "pyrite"),
    ],
    "geology_boundary": [
        ("The boundary between the Earth's crust and mantle is the",
         "Mohorovicic"),
        ("The supercontinent that preceded Pangaea in the late "
         "Precambrian was", "Rodinia"),
        ("The geological period in which dinosaurs first appeared is the",
         "Triassic"),
        ("The geological period immediately before the Cambrian is the",
         "Ediacaran"),
    ],
    "sea_strait_less_cited": [
        ("The strait separating Australia from Tasmania is the Bass",
         "Strait"),
        ("The strait separating Sicily from mainland Italy is the Strait "
         "of", "Messina"),
        ("The strait separating Sri Lanka from India is the Palk",
         "Strait"),
        ("The strait connecting the Black Sea to the Sea of Marmara is "
         "the", "Bosphorus"),
    ],
    "peninsula_region": [
        ("The peninsula shared by Spain and Portugal is the", "Iberian"),
        ("The peninsula containing Denmark is the", "Jutland"),
        ("The peninsula containing Vietnam, Laos and Cambodia is",
         "Indochina"),
        ("The peninsula containing Mumbai's western coast region is the",
         "Deccan"),
    ],
    "sports_record_less_cited": [
        ("The number of players on a water polo team in the pool is",
         "seven"),
        ("The number of frames on a standard bowling scorecard is", "ten"),
        ("The length in metres of an Olympic swimming pool is", "fifty"),
        ("The number of holes in a full round of golf is", "eighteen"),
        ("The number of squares on a Go board's side is", "nineteen"),
    ],
    "instrument_family": [
        ("The instrument family to which the cor anglais belongs is the",
         "woodwind"),
        ("The lowest-pitched member of the standard string quartet is the",
         "cello"),
        ("The keyboard instrument whose strings are plucked, not struck, "
         "is the", "harpsichord"),
        ("The brass instrument with a slide instead of valves is the",
         "trombone"),
    ],
    "cuisine_origin": [
        ("The country where the dish paella originated is", "Spain"),
        ("The country where the dish pho originated is", "Vietnam"),
        ("The country where the dish moussaka originated is", "Greece"),
        ("The country where the dish ceviche is most associated with is",
         "Peru"),
    ],
    "wine_region": [
        ("The French region famous for Sauternes dessert wine is",
         "Bordeaux"),
        ("The Italian region that produces Chianti is", "Tuscany"),
        ("The Spanish region famous for sherry is", "Andalusia"),
        ("The Portuguese river valley famous for port wine is the",
         "Douro"),
    ],
    "organization_founding": [
        ("The city where the United Nations was founded in 1945 is San",
         "Francisco"),
        ("The city that hosts the International Court of Justice is The",
         "Hague"),
        ("The city that hosts the headquarters of the World Health "
         "Organization is", "Geneva"),
        ("The city that hosts the headquarters of OPEC is", "Vienna"),
    ],
    "bird_mammal_specific": [
        ("The only bird known to fly backwards is the", "hummingbird"),
        ("The largest species of penguin is the", "emperor"),
        ("The mammal with the longest gestation period is the",
         "elephant"),
        ("The only venomous primate is the slow", "loris"),
        ("The fastest fish in the ocean is the", "sailfish"),
    ],
    "plant_botany": [
        ("The tallest species of tree on Earth is the coast", "redwood"),
        ("The plant genus that yields natural rubber is", "Hevea"),
        ("The spice derived from the crocus flower is", "saffron"),
        ("The plant from which tequila is made is the blue", "agave"),
    ],
    "computing_history": [
        ("The mathematician who described the first algorithm for a "
         "mechanical computer was Ada", "Lovelace"),
        ("The machine Alan Turing helped design to break Enigma was the",
         "Bombe"),
        ("The first electronic general-purpose computer, unveiled in 1946, "
         "was", "ENIAC"),
        ("The company that produced the first commercial microprocessor "
         "was", "Intel"),
    ],
}


# ---------------------------------------------------------------------
# v3 INCREMENT (2026-07-28, VM7). v2 scored 66/212 in the [-9,-1] window
# on Think — short of the prereg floor (n>=90 across >=30 families).
# The scored pool showed WHICH SHAPES stay hard for a 32B model:
#   productive (>=60% in window): second-order superlatives, classification
#     /category relations (language family, script, instrument family),
#     competing-alternative sets (currencies, moons, missions), derived
#     unit names, numeric records.
#   barren (0 in window): famous-proper-noun recall — chemical symbols,
#     capitals, canonical authors/composers/painters, ancient cities,
#     treaties. The model has these memorized at ceiling regardless of how
#     "obscure" they feel to a human author.
# v3 therefore ADDS families in productive shapes only. v2 families are
# left untouched (superseding by addition; v2's scoring stays reproducible).
FAMILIES_V3: dict[str, list[tuple[str, str]]] = {
    "river_third_order": [
        ("The third-longest river in Africa is the", "Niger"),
        ("The longest river in Germany is the", "Rhine"),
        ("The longest river in Poland is the", "Vistula"),
        ("The longest river in Portugal is the", "Tagus"),
        ("The longest river in Myanmar is the", "Irrawaddy"),
    ],
    "second_city": [
        ("The second-largest city in Australia by population is",
         "Melbourne"),
        ("The second-largest city in Canada by population is",
         "Montreal"),
        ("The second-largest city in Spain by population is",
         "Barcelona"),
        ("The second-largest city in Turkey by population is", "Ankara"),
        ("The second-largest city in Vietnam by population is", "Hanoi"),
    ],
    "animal_class": [
        ("The taxonomic class to which a newt belongs is", "Amphibia"),
        ("The taxonomic class to which a starfish belongs is",
         "Asteroidea"),
        ("The taxonomic class to which an octopus belongs is",
         "Cephalopoda"),
        ("The taxonomic class to which a centipede belongs is",
         "Chilopoda"),
        ("The taxonomic order to which bats belong is", "Chiroptera"),
    ],
    "plant_family": [
        ("The plant family to which the tomato belongs is the",
         "Solanaceae"),
        ("The plant family to which wheat belongs is the", "Poaceae"),
        ("The plant family to which the pea belongs is the", "Fabaceae"),
        ("The plant family to which the sunflower belongs is the",
         "Asteraceae"),
    ],
    "language_family_2": [
        ("The language family to which Georgian belongs is",
         "Kartvelian"),
        ("The language family to which Turkish belongs is", "Turkic"),
        ("The language family to which Mongolian belongs is", "Mongolic"),
        ("The language family to which Vietnamese belongs is",
         "Austroasiatic"),
        ("The language family to which Malagasy belongs is",
         "Austronesian"),
    ],
    "script_2": [
        ("The script used to write the Georgian language is",
         "Mkhedruli"),
        ("The script used to write the Thai language derives from",
         "Khmer"),
        ("The script used to write Old Norse inscriptions is", "runic"),
        ("The script used to write the Cherokee language is a",
         "syllabary"),
    ],
    "stellar_classification": [
        ("The spectral class of the hottest main-sequence stars is",
         "O"),
        ("The spectral class of the Sun is", "G"),
        ("The stellar remnant left by a low-mass star is a white",
         "dwarf"),
        ("The class of variable star used as a distance indicator is the "
         "Cepheid", "variable"),
        ("The stage a star enters after leaving the main sequence is the "
         "red", "giant"),
    ],
    "moons_2": [
        ("The largest moon of Pluto is", "Charon"),
        ("The two moons of Mars are Phobos and", "Deimos"),
        ("The Galilean moon with the most volcanic activity is", "Io"),
        ("The Galilean moon with the thickest ice shell and a suspected "
         "ocean is", "Europa"),
    ],
    "mission_2": [
        ("The mission that returned samples from the asteroid Ryugu was "
         "Hayabusa", "2"),
        ("The mission that first soft-landed on the far side of the Moon "
         "was Chang'e", "4"),
        ("The telescope launched in 2021 to observe in the infrared is "
         "the James Webb Space", "Telescope"),
        ("The probe that entered Jupiter's atmosphere in 1995 was carried "
         "by", "Galileo"),
    ],
    "unit_non_si": [
        ("The unit of pressure equal to 100 kilopascals is the", "bar"),
        ("The unit of energy used in atomic physics is the electron",
         "volt"),
        ("The unit of distance equal to about 3.26 light years is the",
         "parsec"),
        ("The unit of sound intensity level is the", "decibel"),
        ("The unit of viscosity in the CGS system is the", "poise"),
    ],
    "si_prefix": [
        ("The SI prefix denoting ten to the power of minus twelve is",
         "pico"),
        ("The SI prefix denoting ten to the power of fifteen is", "peta"),
        ("The SI prefix denoting ten to the power of minus fifteen is",
         "femto"),
        ("The SI prefix denoting ten to the power of eighteen is", "exa"),
    ],
    "compound_class": [
        ("The class of organic compound containing a carboxyl group is "
         "the carboxylic", "acids"),
        ("The class of organic compound with the general formula CnH2n is "
         "the", "alkenes"),
        ("The class of compound formed from an acid and an alcohol is an",
         "ester"),
        ("The class of organic compound containing a nitrogen atom bonded "
         "to carbon chains is an", "amine"),
    ],
    "rock_classification": [
        ("The rock type formed by cooling magma is", "igneous"),
        ("The rock type formed by heat and pressure on existing rock is",
         "metamorphic"),
        ("The metamorphic equivalent of limestone is", "marble"),
        ("The metamorphic equivalent of shale is", "slate"),
        ("The igneous rock that forms most of the ocean floor is",
         "basalt"),
    ],
    "geological_order": [
        ("The geological period immediately following the Jurassic is the",
         "Cretaceous"),
        ("The geological period immediately preceding the Devonian is the",
         "Silurian"),
        ("The geological epoch immediately preceding the Holocene is the",
         "Pleistocene"),
        ("The geological era immediately preceding the Mesozoic is the",
         "Paleozoic"),
    ],
    "music_theory": [
        ("The musical interval spanning seven semitones is a perfect",
         "fifth"),
        ("The key signature with four sharps is the key of", "E"),
        ("The relative minor of C major is", "A"),
        ("The tempo marking meaning 'walking pace' is", "andante"),
        ("The musical form built on a repeating bass line is the",
         "passacaglia"),
    ],
    "art_movement_order": [
        ("The art movement that immediately followed Impressionism is",
         "Post"),
        ("The art movement founded by Andre Breton in 1924 is",
         "Surrealism"),
        ("The art movement of Malevich's geometric abstraction is",
         "Suprematism"),
        ("The architectural style preceding Gothic in medieval Europe is",
         "Romanesque"),
    ],
    "philosophy_school": [
        ("The philosophical school founded by Zeno of Citium is",
         "Stoicism"),
        ("The philosophical school holding that truth is what works is",
         "pragmatism"),
        ("The school of thought associated with Ockham's razor is",
         "nominalism"),
        ("The ethical theory judging acts by their consequences is",
         "consequentialism"),
    ],
    "government_system": [
        ("The system of government in which power is divided between "
         "national and regional levels is", "federalism"),
        ("The parliamentary system's head of government in Germany is "
         "called the", "Chancellor"),
        ("The voting system in which candidates are ranked and votes "
         "transferred is the single transferable", "vote"),
        ("The doctrine that courts may invalidate legislation is judicial",
         "review"),
    ],
    "economics_measure": [
        ("The index measuring price changes for a basket of consumer "
         "goods is the consumer price", "index"),
        ("The measure of an economy's output per person is GDP per",
         "capita"),
        ("The unemployment that persists due to skill mismatch is called",
         "structural"),
        ("The curve relating unemployment to inflation is the", "Phillips"),
    ],
    "network_computing": [
        ("The default network port for HTTPS traffic is", "443"),
        ("The protocol that resolves domain names to addresses is", "DNS"),
        ("The layer of the OSI model that handles routing is the",
         "network"),
        ("The algorithm most used for public-key exchange over insecure "
         "channels is Diffie", "Hellman"),
        ("The data structure with amortized constant lookup by key is the "
         "hash", "table"),
    ],
    "algorithm_complexity": [
        ("The average time complexity of quicksort is O(n log", "n"),
        ("The sorting algorithm that guarantees O(n log n) worst case by "
         "merging is", "merge"),
        ("The graph algorithm finding shortest paths from one source with "
         "non-negative weights is", "Dijkstra"),
        ("The technique of storing subproblem results to avoid recompute "
         "is", "memoization"),
    ],
    "medicine_drug_class": [
        ("The drug class that lowers cholesterol by inhibiting HMG-CoA "
         "reductase is the", "statins"),
        ("The drug class used to treat depression by blocking serotonin "
         "reuptake is the", "SSRIs"),
        ("The drug class that reduces blood pressure by blocking "
         "angiotensin conversion is the ACE", "inhibitors"),
        ("The class of drug that reduces inflammation without steroids is "
         "the", "NSAIDs"),
    ],
    "physiology_process": [
        ("The process by which the kidney returns useful solutes to the "
         "blood is", "reabsorption"),
        ("The process by which cells engulf large particles is",
         "phagocytosis"),
        ("The process by which mRNA is made from DNA is", "transcription"),
        ("The process of programmed cell death is", "apoptosis"),
        ("The process by which muscle produces energy without oxygen is "
         "anaerobic", "respiration"),
    ],
    "border_specific": [
        ("The only country bordering both Portugal and France is", "Spain"),
        ("The country that borders both Poland and Hungary is",
         "Slovakia"),
        ("The country that borders both Iran and China is", "Afghanistan"),
        ("The two countries bordering Lesotho number exactly", "one"),
    ],
    "sports_numeric_2": [
        ("The number of points awarded for a try in rugby union is",
         "five"),
        ("The number of players on a volleyball team on court is", "six"),
        ("The maximum break in snooker without fouls is", "147"),
        ("The number of rounds in a championship boxing bout is",
         "twelve"),
        ("The distance in metres of an Olympic steeplechase is", "3000"),
    ],
    "measure_numeric": [
        ("The number of degrees in each interior angle of a regular "
         "hexagon is", "120"),
        ("The number of bones in the adult human body is", "206"),
        ("The number of chromosomes in a human somatic cell is",
         "forty"),
        ("The number of elements in the seventh period of the periodic "
         "table is", "32"),
    ],
    "material_composition": [
        ("The alloy of copper and tin is", "bronze"),
        ("The alloy of copper and zinc is", "brass"),
        ("The alloy of iron with chromium that resists corrosion is "
         "stainless", "steel"),
        ("The ceramic material used in spacecraft heat shields is often "
         "silica", "aerogel"),
    ],
    "instrument_family_2": [
        ("The instrument family to which the celesta belongs is the",
         "percussion"),
        ("The highest-pitched member of the standard brass section is the",
         "trumpet"),
        ("The double-reed instrument pitched below the oboe is the",
         "bassoon"),
        ("The string instrument played with a bow and held between the "
         "knees historically is the viola da", "gamba"),
    ],
    "linguistics_term": [
        ("The smallest unit of sound that distinguishes meaning is the",
         "phoneme"),
        ("The smallest unit of meaning in a word is the", "morpheme"),
        ("The study of meaning in language is", "semantics"),
        ("The word order typical of Japanese sentences is subject object",
         "verb"),
        ("The consonant produced by stopping airflow completely is a",
         "plosive"),
    ],
    "cartography_time": [
        ("The line of longitude at zero degrees is the prime", "meridian"),
        ("The map projection preserving angles but distorting area is the",
         "Mercator"),
        ("The latitude of the Tropic of Cancer is about 23.5 degrees",
         "north"),
        ("The number of standard time zones spanning the globe is",
         "24"),
    ],
}


# CANONICAL FAMILY MAP — v3 deliberately extended several v2 templates
# with fresh items (`language_family_2` holds more of the same relation as
# `language_family`). Those are NOT independent clusters: pooling them as
# two families would be the pseudo-replication the addendum flagged in the
# SQL battery. Analysis must cluster on the CANONICAL name.
CANONICAL_FAMILY: dict[str, str] = {
    "language_family_2": "language_family",
    "script_2": "script_writing_system",
    "moons_2": "moons_planetary",
    "mission_2": "spacecraft_mission",
    "sports_numeric_2": "sports_record_less_cited",
    "instrument_family_2": "instrument_family",
    # ordinal-superlative river/mountain prompts share one relation schema
    "river_third_order": "river_second_order",
    # ---- v4/v5 re-authorings caught by the G5 duplicate-fact audit ------
    # Each of these was authored as a NEW family but expresses the SAME
    # relation over the SAME surface template as an existing one, and the
    # audit found items that are literally the same fact on both sides
    # (e.g. "The second-longest river in Africa is the" appears in both).
    # Treating them as independent clusters would let ONE fact land in both
    # the confirmatory and the replication partition.
    "second_longest_river_country": "river_second_order",
    "second_highest_peak": "mountain_second_order",
    "second_largest_lake": "lake_specific",
    "philosophical_work_author": "philosopher_work",
    "economic_work_author": "economics_concept_person",
    "language_family_membership": "language_family",
    "sporting_count": "sports_record_less_cited",
    "biological_process_name": "physiology_process",
    "anatomical_count": "measure_numeric",
}

# COARSER CLUSTERING, offered so the estimand choice is explicit rather
# than implicit (nextsteps_2_2 §9.1). Two families can share a RELATION
# (work -> creator) while using different surface templates and disjoint
# knowledge domains: "who wrote Leviathan" and "who wrote Silent Spring"
# share the relation but not the facts. `canonical_family` keeps them
# apart; `relation_group` pools them. The preregistration names
# canonical_family as primary and relation_group as prespecified
# sensitivity, so a reviewer who thinks the pooling should be coarser can
# read that analysis instead of doubting the primary.
RELATION_GROUP: dict[str, str] = {
    # work / concept -> its creator
    "philosopher_work": "work_to_creator",
    "author_less_canonical": "work_to_creator",
    "playwright_work": "work_to_creator",
    "composer_work_less_cited": "work_to_creator",
    "painter_work_less_cited": "work_to_creator",
    "architect_building": "work_to_creator",
    "scientific_work_author": "work_to_creator",
    "political_work_author": "work_to_creator",
    "sociological_concept_author": "work_to_creator",
    "economics_concept_person": "work_to_creator",
    "mathematician_theorem": "work_to_creator",
    "physicist_named_effect": "work_to_creator",
    "programming_language_creator": "work_to_creator",
    "algorithm_inventor": "work_to_creator",
    "medical_discoverer": "work_to_creator",
    # entity -> the category it belongs to
    "language_family": "entity_to_category",
    "script_writing_system": "entity_to_category",
    "animal_class": "entity_to_category",
    "plant_family": "entity_to_category",
    "compound_class": "entity_to_category",
    "rock_classification": "entity_to_category",
    "rock_origin_class": "entity_to_category",
    "crystal_system": "entity_to_category",
    "chemical_bond_type": "entity_to_category",
    "pathogen_type": "entity_to_category",
    "star_spectral_class": "entity_to_category",
    "programming_paradigm": "entity_to_category",
    "protocol_layer": "entity_to_category",
    "language_word_order": "entity_to_category",
    "art_movement_order": "entity_to_category",
    "architecture_period_feature": "entity_to_category",
    "philosophy_school": "entity_to_category",
    "medical_specialty_scope": "entity_to_category",
    "instrument_family": "entity_to_category",
    # ordinal / second-order superlative
    "river_second_order": "ordinal_superlative",
    "mountain_second_order": "ordinal_superlative",
    "lake_specific": "ordinal_superlative",
    "second_city": "ordinal_superlative",
    "second_most_populous": "ordinal_superlative",
    "peninsula_region": "ordinal_superlative",
    "star_brightest_constellation": "ordinal_superlative",
    # counts of structured objects
    "measure_numeric": "structured_count",
    "polygon_solid_count": "structured_count",
    "poetic_form_length": "structured_count",
    "moons_planetary": "structured_count",
    "sports_record_less_cited": "structured_count",
    "music_interval_count": "structured_count",
    "musical_ensemble_size": "structured_count",
    "chemistry_count": "structured_count",
    # names of processes and mechanisms
    "physiology_process": "process_name",
    "geological_process_name": "process_name",
    "chemical_process_name": "process_name",
    "physics_process_name": "process_name",
    "psychological_process_name": "process_name",
    "linguistic_process_name": "process_name",
    "computing_process_name": "process_name",
    "geographic_process_name": "process_name",
    "ecological_process_name": "process_name",
    "enzyme_process": "process_name",
    "optical_phenomenon_name": "process_name",
    # terminology: definition -> its technical term
    "cognitive_bias_name": "definition_to_term",
    "economic_term_definition": "definition_to_term",
    "logical_fallacy_name": "definition_to_term",
    "statistical_measure_definition": "definition_to_term",
    "geometry_term_definition": "definition_to_term",
    "governance_term_definition": "definition_to_term",
    "literary_device_name": "definition_to_term",
    "art_technique_name": "definition_to_term",
    "grammatical_case_function": "definition_to_term",
    "linguistics_term": "definition_to_term",
    "study_of_field_name": "definition_to_term",
    "phobia_or_philia_object": "definition_to_term",
    "measurement_instrument": "definition_to_term",
    "measurement_scale_name": "definition_to_term",
    "unit_measures_quantity": "definition_to_term",
    "si_unit_derived": "definition_to_term",
}


def relation_group(family: str) -> str:
    """Coarser than canonical_family: pools families that share a relation
    but not a surface template or knowledge domain. Prespecified
    SENSITIVITY clustering, never the primary."""
    return RELATION_GROUP.get(canonical_family(family),
                              canonical_family(family))


def canonical_family(family: str) -> str:
    return CANONICAL_FAMILY.get(family, family)


def pool_rows(version: str = "v2") -> list[dict]:
    """Flat candidate rows: {prompt, answer, family, pool}. Prompt carries
    the 'Fact: ' prefix (v1 scoring convention); answer carries a leading
    space (battery variant convention adds capitalization variants).
    version 'v2' = the original 45 families; 'v3' = the increment only;
    'all' = both (the Stage-3 bank)."""
    src = {"v2": [("v2", FAMILIES)], "v3": [("v3", FAMILIES_V3)],
           "all": [("v2", FAMILIES), ("v3", FAMILIES_V3)]}[version]
    rows = []
    for tag, fams in src:
        for family, items in fams.items():
            for prompt, answer in items:
                rows.append({"prompt": f"Fact: {prompt}",
                             "answer": f" {answer}", "family": family,
                             "pool": tag})
    return rows


def summary(version: str = "v2") -> dict:
    rows = pool_rows(version)
    fams = sorted({r["family"] for r in rows})
    return {"version": version, "n_candidates": len(rows),
            "n_families": len(fams),
            "items_per_family": {f: sum(1 for r in rows if r["family"] == f)
                                 for f in fams}}
