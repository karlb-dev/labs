# Stage-3 bank, v4 increment — the D5 expansion to a family-disjoint bank.
#
# WHY. The scored v2+v3 bank clears a single n>=90 / 30-family floor, but
# only **18 canonical families** have >=3 in-window items. A confirmatory
# and a replication partition that are independent at the FAMILY level
# need >=30 such families EACH, so the bank must roughly triple its count
# of well-populated families. Splitting items inside a shared template
# would give an item holdout, not an independent replication.
#
# TARGETING. v4 authors only in the shapes the scored pool showed stay
# hard for a 32B model, and deliberately avoids the shape that failed
# (famous-proper-noun recall went 0/16 families in v2):
#   taxonomic/classification relations  language_family 9/10, animal_class
#                                       4/5, philosophy_school 4/4,
#                                       script_writing_system 5/8
#   second-order superlatives           river_second_order 7/10,
#                                       peninsula_region 3/4, second_city 3/5
#   competing-alternative sets          spacecraft_mission 6/8,
#                                       sports_record 6/10
#   derived units / numerics            si_unit_derived 3/6, moons 5/9
#   attribution of a NAMED concept      programming_language_creator 4/5,
#                                       mathematician_theorem 3/5
#   composition / process names         material_composition 3/4,
#                                       physiology_process 4/5
#
# Every family here is a NEW canonical relation, not an extension of an
# existing one — extensions would collapse under CANONICAL_FAMILY and buy
# no independent clusters, which is the whole point of the expansion.
#
# 8 items per family so that a ~50% in-window rate still leaves >=3.
# Answers are single words to stay inside the battery's frozen alias rule
# ({" ans", " Ans"}).
from __future__ import annotations

FAMILIES_V4: dict[str, list[tuple[str, str]]] = {
    # ---- taxonomic / classification -------------------------------------
    "language_word_order": [
        ("The basic word order of Japanese is subject", "object"),
        ("The basic word order of Turkish is subject", "object"),
        ("The basic word order of Korean is subject", "object"),
        ("The basic word order of Hindi is subject", "object"),
        ("The basic word order of Irish is verb", "subject"),
        ("The basic word order of Welsh is verb", "subject"),
        ("The basic word order of Arabic in its classical form is verb", "subject"),
        ("The basic word order of Malagasy is verb", "object"),
    ],
    "star_spectral_class": [
        ("The spectral class of Betelgeuse is", "M"),
        ("The spectral class of Rigel is", "B"),
        ("The spectral class of Vega is", "A"),
        ("The spectral class of Sirius A is", "A"),
        ("The spectral class of Arcturus is", "K"),
        ("The spectral class of Aldebaran is", "K"),
        ("The spectral class of Procyon A is", "F"),
        ("The spectral class of Antares is", "M"),
    ],
    "crystal_system": [
        ("The crystal system of halite is", "cubic"),
        ("The crystal system of quartz is", "trigonal"),
        ("The crystal system of gypsum is", "monoclinic"),
        ("The crystal system of zircon is", "tetragonal"),
        ("The crystal system of olivine is", "orthorhombic"),
        ("The crystal system of beryl is", "hexagonal"),
        ("The crystal system of pyrite is", "cubic"),
        ("The crystal system of calcite is", "trigonal"),
    ],
    "chemical_bond_type": [
        ("The type of bonding in sodium chloride is", "ionic"),
        ("The type of bonding in methane is", "covalent"),
        ("The type of bonding in brass is", "metallic"),
        ("The type of bonding in magnesium oxide is", "ionic"),
        ("The type of bonding in carbon dioxide is", "covalent"),
        ("The type of bonding in tungsten is", "metallic"),
        ("The type of bonding in calcium fluoride is", "ionic"),
        ("The type of bonding in ammonia is", "covalent"),
    ],
    "rock_origin_class": [
        ("Basalt is classified as an igneous rock formed by", "extrusive"),
        ("Granite is classified as an igneous rock formed by", "intrusive"),
        ("Gabbro is classified as an igneous rock formed by", "intrusive"),
        ("Rhyolite is classified as an igneous rock formed by", "extrusive"),
        ("Obsidian is classified as an igneous rock formed by", "extrusive"),
        ("Diorite is classified as an igneous rock formed by", "intrusive"),
        ("Andesite is classified as an igneous rock formed by", "extrusive"),
        ("Peridotite is classified as an igneous rock formed by", "intrusive"),
    ],
    "pathogen_type": [
        ("The disease malaria is caused by an organism that is a", "parasite"),
        ("The disease tuberculosis is caused by an organism that is a", "bacterium"),
        ("The disease influenza is caused by an organism that is a", "virus"),
        ("The disease cholera is caused by an organism that is a", "bacterium"),
        ("The disease ringworm is caused by an organism that is a", "fungus"),
        ("The disease giardiasis is caused by an organism that is a", "parasite"),
        ("The disease shingles is caused by an organism that is a", "virus"),
        ("The disease tetanus is caused by an organism that is a", "bacterium"),
    ],
    "hormone_source_gland": [
        ("The gland that secretes insulin is the", "pancreas"),
        ("The gland that secretes thyroxine is the", "thyroid"),
        ("The gland that secretes cortisol is the", "adrenal"),
        ("The gland that secretes melatonin is the", "pineal"),
        ("The gland that secretes growth hormone is the", "pituitary"),
        ("The gland that secretes calcitonin is the", "thyroid"),
        ("The gland that secretes adrenaline is the", "adrenal"),
        ("The gland that secretes parathyroid hormone is the", "parathyroid"),
    ],
    "programming_paradigm": [
        ("The programming paradigm that Haskell is best known for is", "functional"),
        ("The programming paradigm that Prolog is best known for is", "logic"),
        ("The programming paradigm that Smalltalk is best known for is", "object"),
        ("The programming paradigm that Erlang is best known for is", "functional"),
        ("The programming paradigm that Datalog is best known for is", "logic"),
        ("The programming paradigm that OCaml is best known for is", "functional"),
        ("The programming paradigm that Simula pioneered is", "object"),
        ("The programming paradigm that Mercury is best known for is", "logic"),
    ],
    "protocol_layer": [
        ("In the OSI model, the layer at which TCP operates is the", "transport"),
        ("In the OSI model, the layer at which IP operates is the", "network"),
        ("In the OSI model, the layer at which Ethernet operates is the", "data"),
        ("In the OSI model, the layer at which HTTP operates is the", "application"),
        ("In the OSI model, the layer at which UDP operates is the", "transport"),
        ("In the OSI model, the layer at which ARP is usually placed is the", "link"),
        ("In the OSI model, the layer at which SMTP operates is the", "application"),
        ("In the OSI model, the layer at which ICMP is placed is the", "network"),
    ],
    # ---- second-order superlatives ----------------------------------------
    "second_highest_peak": [
        ("The second-highest mountain in Africa is", "Kenya"),
        ("The second-highest mountain in South America is", "Ojos"),
        ("The second-highest mountain in North America is", "Logan"),
        ("The second-highest mountain in Europe is", "Dykh"),
        ("The second-highest mountain in Japan is", "Kita"),
        ("The second-highest mountain in the Alps is", "Dufourspitze"),
        ("The second-highest volcano in the world is", "Ojos"),
        ("The second-highest mountain in Antarctica is", "Tyree"),
    ],
    "second_largest_lake": [
        ("The second-largest lake in Africa by area is Lake", "Victoria"),
        ("The second-largest of the Great Lakes by area is Lake", "Huron"),
        ("The second-largest lake in South America by area is Lake", "Maracaibo"),
        ("The second-largest freshwater lake in the world by volume is Lake", "Tanganyika"),
        ("The second-largest lake in Canada entirely within its borders is Great", "Bear"),
        ("The second-deepest lake in the world is Lake", "Tanganyika"),
        ("The second-largest lake in Europe by area is Lake", "Onega"),
        ("The second-largest saltwater lake in the world is the", "Caspian"),
    ],
    "second_longest_river_country": [
        ("The second-longest river in China is the", "Yellow"),
        ("The second-longest river in the United States by length is the", "Mississippi"),
        ("The second-longest river in India is the", "Godavari"),
        ("The second-longest river in Russia is the", "Ob"),
        ("The second-longest river in Germany is the", "Elbe"),
        ("The second-longest river in France is the", "Rhone"),
        ("The second-longest river in Australia is the", "Darling"),
        ("The second-longest river in Africa is the", "Congo"),
    ],
    "second_most_populous": [
        ("The second-most populous country in Africa is", "Ethiopia"),
        ("The second-most populous country in South America is", "Colombia"),
        ("The second-most populous country in Europe is", "Germany"),
        ("The second-most populous city in Japan is", "Yokohama"),
        ("The second-most populous country in North America is", "Mexico"),
        ("The second-most populous country in Southeast Asia is", "Philippines"),
        ("The second-most spoken native language in the world is", "Spanish"),
        ("The second-most populous city in Canada is", "Montreal"),
    ],
    # ---- competing alternatives ------------------------------------------
    "first_of_pair_exploration": [
        ("The first person to reach the South Pole was", "Amundsen"),
        ("The first person to fly solo nonstop across the Atlantic was", "Lindbergh"),
        ("The first woman to fly solo across the Atlantic was", "Earhart"),
        ("The first person to sail single-handed around the world was", "Slocum"),
        ("The first people to reach the summit of Everest included Tenzing and", "Hillary"),
        ("The first person to reach the deepest point of the ocean in 1960 with Walsh was", "Piccard"),
        ("The first person to walk in space was", "Leonov"),
        ("The first person to fly faster than sound was", "Yeager"),
    ],
    "nobel_shared_discovery": [
        ("The element polonium was discovered by Pierre Curie and", "Marie"),
        ("The structure of DNA was published by Francis Crick and James", "Watson"),
        ("The neutron was discovered by James", "Chadwick"),
        ("The positron was discovered by Carl", "Anderson"),
        ("Penicillin was discovered by Alexander", "Fleming"),
        ("The electron was discovered by J.J.", "Thomson"),
        ("Radio waves were first produced in the laboratory by Heinrich", "Hertz"),
        ("The cosmic microwave background was discovered by Penzias and", "Wilson"),
    ],
    "spacecraft_target_body": [
        ("The spacecraft Cassini was sent to study", "Saturn"),
        ("The spacecraft Galileo was sent to study", "Jupiter"),
        ("The spacecraft Magellan was sent to study", "Venus"),
        ("The spacecraft MESSENGER was sent to study", "Mercury"),
        ("The spacecraft Juno was sent to study", "Jupiter"),
        ("The spacecraft Rosetta was sent to study a", "comet"),
        ("The spacecraft Dawn was sent to study Vesta and", "Ceres"),
        ("The spacecraft New Horizons was sent to study", "Pluto"),
    ],
    "olympic_host_city_country": [
        ("The 1964 Summer Olympics were held in the country of", "Japan"),
        ("The 1968 Summer Olympics were held in the country of", "Mexico"),
        ("The 1972 Summer Olympics were held in the country of", "Germany"),
        ("The 1980 Summer Olympics were held in the country of", "Russia"),
        ("The 1988 Summer Olympics were held in the country of", "Korea"),
        ("The 1992 Summer Olympics were held in the country of", "Spain"),
        ("The 2004 Summer Olympics were held in the country of", "Greece"),
        ("The 2016 Summer Olympics were held in the country of", "Brazil"),
    ],
    # ---- derived units, constants, numerics -------------------------------
    "unit_measures_quantity": [
        ("The physical quantity measured in pascals is", "pressure"),
        ("The physical quantity measured in teslas is magnetic", "field"),
        ("The physical quantity measured in siemens is", "conductance"),
        ("The physical quantity measured in becquerels is", "radioactivity"),
        ("The physical quantity measured in candelas is luminous", "intensity"),
        ("The physical quantity measured in farads is", "capacitance"),
        ("The physical quantity measured in henries is", "inductance"),
        ("The physical quantity measured in webers is magnetic", "flux"),
    ],
    "currency_subunit": [
        ("The subunit of the Indian rupee is the", "paisa"),
        ("The subunit of the Japanese yen is the", "sen"),
        ("The subunit of the Russian rouble is the", "kopeck"),
        ("The subunit of the Swedish krona is the", "ore"),
        ("The subunit of the Polish zloty is the", "grosz"),
        ("The subunit of the Danish krone is the", "ore"),
        ("The subunit of the South African rand is the", "cent"),
        ("The subunit of the Thai baht is the", "satang"),
    ],
    "polygon_solid_count": [
        ("The number of faces on a regular dodecahedron is", "twelve"),
        ("The number of faces on a regular icosahedron is", "twenty"),
        ("The number of edges on a cube is", "twelve"),
        ("The number of vertices on a regular octahedron is", "six"),
        ("The number of edges on a regular tetrahedron is", "six"),
        ("The number of sides in a heptagon is", "seven"),
        ("The number of sides in a nonagon is", "nine"),
        ("The number of sides in a dodecagon is", "twelve"),
    ],
    "poetic_form_length": [
        ("The number of lines in a sonnet is", "fourteen"),
        ("The number of lines in a limerick is", "five"),
        ("The number of lines in a haiku is", "three"),
        ("The number of lines in a villanelle is", "nineteen"),
        ("The number of lines in a couplet is", "two"),
        ("The number of lines in a sestina excluding the envoi is", "thirty"),
        ("The number of lines in a tercet is", "three"),
        ("The number of lines in a quatrain is", "four"),
    ],
    # ---- attribution of a named concept -----------------------------------
    "named_law_field": [
        ("Ohm's law belongs to the field of", "electricity"),
        ("Boyle's law belongs to the study of", "gases"),
        ("Hooke's law describes the behaviour of a", "spring"),
        ("Snell's law describes the behaviour of", "light"),
        ("Fick's law describes the process of", "diffusion"),
        ("Hubble's law describes the expansion of the", "universe"),
        ("Faraday's law describes electromagnetic", "induction"),
        ("Bernoulli's principle describes the behaviour of a moving", "fluid"),
    ],
    "algorithm_inventor": [
        ("The shortest-path algorithm named after its inventor Edsger is", "Dijkstra"),
        ("The sorting algorithm invented by Tony Hoare is called", "quicksort"),
        ("The data compression code invented by David is called", "Huffman"),
        ("The public-key algorithm named for Rivest, Shamir and", "Adleman"),
        ("The fast Fourier transform is commonly named for Cooley and", "Tukey"),
        ("The minimum spanning tree algorithm named for Joseph is", "Kruskal"),
        ("The string search algorithm named for Knuth, Morris and", "Pratt"),
        ("The network flow algorithm named for Ford and", "Fulkerson"),
    ],
    "philosophical_work_author": [
        ("The work Critique of Pure Reason was written by", "Kant"),
        ("The work Being and Time was written by", "Heidegger"),
        ("The work Leviathan was written by", "Hobbes"),
        ("The work The Social Contract was written by", "Rousseau"),
        ("The work Tractatus Logico-Philosophicus was written by", "Wittgenstein"),
        ("The work Ethics, written in geometrical order, is by", "Spinoza"),
        ("The work The Structure of Scientific Revolutions was written by", "Kuhn"),
        ("The work A Theory of Justice was written by", "Rawls"),
    ],
    # ---- composition, process and mechanism names -------------------------
    "alloy_main_components": [
        ("The two main metals in bronze are copper and", "tin"),
        ("The two main metals in brass are copper and", "zinc"),
        ("The main metal alloyed with carbon to make steel is", "iron"),
        ("The main metal in pewter is", "tin"),
        ("The two main metals in sterling silver are silver and", "copper"),
        ("The main metal alloyed with copper in cupronickel is", "nickel"),
        ("The main metal in solder traditionally alloyed with lead is", "tin"),
        ("The two main metals in electrum are gold and", "silver"),
    ],
    "biological_process_name": [
        ("The process by which plants lose water vapour through their leaves is", "transpiration"),
        ("The process by which a cell engulfs solid particles is", "phagocytosis"),
        ("The process by which water moves across a membrane is", "osmosis"),
        ("The process by which mRNA is made from DNA is", "transcription"),
        ("The process by which a protein is built from mRNA is", "translation"),
        ("The process by which a cell divides to form gametes is", "meiosis"),
        ("The process by which glucose is broken down without oxygen is", "glycolysis"),
        ("The process by which a tadpole becomes a frog is", "metamorphosis"),
    ],
    "geological_process_name": [
        ("The process by which rock is broken down in place by weather is", "weathering"),
        ("The process by which sediment is carried away by water or wind is", "erosion"),
        ("The process by which sediment is dropped and accumulates is", "deposition"),
        ("The process by which sediment hardens into rock is", "lithification"),
        ("The process by which one tectonic plate slides beneath another is", "subduction"),
        ("The process by which rock changes form under heat and pressure is", "metamorphism"),
        ("The process by which magma cools and hardens is", "crystallization"),
        ("The process by which soil creeps slowly downhill is", "solifluction"),
    ],
    "optical_phenomenon_name": [
        ("The bending of light as it passes between media is called", "refraction"),
        ("The spreading of light as it passes through a narrow slit is called", "diffraction"),
        ("The splitting of white light into colours by a prism is called", "dispersion"),
        ("The bouncing of light off a surface is called", "reflection"),
        ("The restriction of light waves to one plane is called", "polarization"),
        ("The reinforcement and cancellation of overlapping waves is called", "interference"),
        ("The scattering of light by very small particles is called Rayleigh", "scattering"),
        ("The complete reflection of light inside a fibre is called total internal", "reflection"),
    ],
    "economic_term_definition": [
        ("A sustained fall in the general price level is called", "deflation"),
        ("A period of rising prices combined with stagnation is called", "stagflation"),
        ("A market with a single seller is called a", "monopoly"),
        ("A market with a single buyer is called a", "monopsony"),
        ("A market dominated by a few sellers is called an", "oligopoly"),
        ("The value of the next best alternative forgone is called opportunity", "cost"),
        ("A tax that takes a larger share from low incomes is called", "regressive"),
        ("The additional satisfaction from one more unit is called marginal", "utility"),
    ],
    "cognitive_bias_name": [
        ("The tendency to favour information confirming existing beliefs is called confirmation", "bias"),
        ("The tendency to rely too heavily on the first piece of information is called", "anchoring"),
        ("The tendency to judge frequency by how easily examples come to mind is called the availability", "heuristic"),
        ("The tendency of unskilled people to overestimate their ability is called the Dunning", "Kruger"),
        ("The tendency to continue a project because of past investment is called the sunk cost", "fallacy"),
        ("The tendency to see past events as having been predictable is called", "hindsight"),
        ("The tendency to conform to the behaviour of a group is called", "conformity"),
        ("The tendency to attribute others' actions to character rather than situation is called the fundamental attribution", "error"),
    ],
    "architecture_period_feature": [
        ("The architectural style characterized by pointed arches and flying buttresses is", "Gothic"),
        ("The architectural style characterized by rounded arches and thick walls is", "Romanesque"),
        ("The architectural style characterized by ornate drama and movement is", "Baroque"),
        ("The architectural style reviving Greek and Roman forms in the Renaissance is", "Classical"),
        ("The architectural style characterized by glass, steel and no ornament is", "Modernist"),
        ("The architectural style characterized by raw exposed concrete is", "Brutalist"),
        ("The architectural style of flowing plant-like ornament around 1900 is Art", "Nouveau"),
        ("The architectural style of geometric luxury in the 1920s is Art", "Deco"),
    ],
    "measurement_instrument": [
        ("The instrument used to measure atmospheric pressure is a", "barometer"),
        ("The instrument used to measure humidity is a", "hygrometer"),
        ("The instrument used to measure earthquakes is a", "seismograph"),
        ("The instrument used to measure wind speed is an", "anemometer"),
        ("The instrument used to measure the angle of stars above the horizon is a", "sextant"),
        ("The instrument used to measure electric current is an", "ammeter"),
        ("The instrument used to measure very small lengths precisely is a", "micrometer"),
        ("The instrument used to measure the density of a liquid is a", "hydrometer"),
    ],
    "collective_noun_animal": [
        ("A group of crows is called a", "murder"),
        ("A group of lions is called a", "pride"),
        ("A group of geese in flight is called a", "skein"),
        ("A group of owls is called a", "parliament"),
        ("A group of ravens is called an", "unkindness"),
        ("A group of larks is called an", "exaltation"),
        ("A group of jellyfish is called a", "smack"),
        ("A group of ferrets is called a", "business"),
    ],
    "body_system_function": [
        ("The body system responsible for producing movement is the", "muscular"),
        ("The body system responsible for filtering blood into urine is the", "urinary"),
        ("The body system responsible for hormone signalling is the", "endocrine"),
        ("The body system responsible for defence against infection is the", "immune"),
        ("The body system responsible for gas exchange is the", "respiratory"),
        ("The body system responsible for support and protection is the", "skeletal"),
        ("The body system responsible for returning fluid from tissues is the", "lymphatic"),
        ("The body system responsible for breaking down food is the", "digestive"),
    ],
}


def rows_v4() -> list[dict]:
    return [{"prompt": f"Fact: {p}", "answer": f" {a}", "family": fam,
             "pool": "v4"}
            for fam, items in FAMILIES_V4.items() for p, a in items]


if __name__ == "__main__":
    r = rows_v4()
    print(f"v4: {len(FAMILIES_V4)} NEW canonical families, {len(r)} candidate "
          f"items ({len(r) / len(FAMILIES_V4):.1f} per family)")
