# Stage-3 bank, v5 increment — closing the gap to a family-disjoint bank.
#
# After v4 the bank holds 40 canonical families with >=3 in-window items
# (18 from v2/v3 + 22 from v4). D5 needs >=60 so that confirmatory and
# replication partitions can each hold >=30 DISJOINT families. v5 targets
# the remaining ~20-25.
#
# TARGETING IS NOW EMPIRICAL, not guessed. v4's per-family yields identify
# exactly which shapes stay hard for a 32B model:
#     8/8  second_longest_river_country, polygon_solid_count,
#          philosophical_work_author, biological_process_name
#     7/8  language_word_order, pathogen_type
#     6/8  poetic_form_length, geological_process_name,
#          architecture_period_feature
# and which collapse to ceiling — the SAME failure mode v2 found:
#     0/8  star_spectral_class, crystal_system, spacecraft_target_body,
#          alloy_main_components   (all famous-entity recall)
#
# The winning shape is consistent: name a PROCESS, COUNT, or CATEGORY that
# the model must compose from structured knowledge, rather than an entity
# it has memorised as a high-frequency completion. v5 authors only in
# those shapes.
from __future__ import annotations

FAMILIES_V5: dict[str, list[tuple[str, str]]] = {
    # ---- process / mechanism names (v4's strongest shape) -----------------
    "chemical_process_name": [
        ("The process by which a liquid changes directly to a solid is", "freezing"),
        ("The process by which a solid changes directly to a gas is", "sublimation"),
        ("The process by which a gas changes directly to a solid is", "deposition"),
        ("The process by which a salt reacts with water to change pH is", "hydrolysis"),
        ("The process by which large molecules are broken by water is", "hydrolysis"),
        ("The process by which two molecules join and release water is", "condensation"),
        ("The process by which an atom loses electrons is", "oxidation"),
        ("The process by which a metal is extracted from ore by heating is", "smelting"),
    ],
    "physics_process_name": [
        ("The transfer of heat by the movement of a fluid is called", "convection"),
        ("The transfer of heat through direct contact is called", "conduction"),
        ("The transfer of heat through empty space is called", "radiation"),
        ("The build-up of large oscillations at a natural frequency is called", "resonance"),
        ("The gradual loss of wave energy passing through a medium is called", "attenuation"),
        ("The change in observed frequency due to relative motion is the", "Doppler"),
        ("The emission of electrons from a metal struck by light is the", "photoelectric"),
        ("The decay of a quantity at a rate proportional to itself is called", "exponential"),
    ],
    "psychological_process_name": [
        ("Learning to associate a neutral stimulus with a reflex is called classical", "conditioning"),
        ("Learning shaped by rewards and punishments is called operant", "conditioning"),
        ("The decline in response to a repeated harmless stimulus is called", "habituation"),
        ("The weakening of a learned response when reinforcement stops is called", "extinction"),
        ("Recalling information without cues is called free", "recall"),
        ("Interference from older memories on newer ones is called proactive", "interference"),
        ("The improved recall of the first items in a list is the primacy", "effect"),
        ("The improved recall of the last items in a list is the recency", "effect"),
    ],
    "linguistic_process_name": [
        ("The process by which two sounds become more alike is called", "assimilation"),
        ("The loss of a sound from the middle of a word is called", "syncope"),
        ("The loss of a final sound from a word is called", "apocope"),
        ("The insertion of an extra sound into a word is called", "epenthesis"),
        ("The swapping of two sounds within a word is called", "metathesis"),
        ("The broadening of a word's meaning over time is called semantic", "widening"),
        ("The process of forming a word from initial letters is called", "acronymy"),
        ("The formation of a shorter word by removing a supposed affix is called back", "formation"),
    ],
    "computing_process_name": [
        ("The technique of storing recently used data for faster access is called", "caching"),
        ("The technique of splitting a task across many machines is called", "distribution"),
        ("The technique of converting an object to a storable byte stream is called", "serialization"),
        ("The technique of running several processes on one processor by switching is called", "multitasking"),
        ("The technique of reclaiming unused memory automatically is called garbage", "collection"),
        ("The technique of moving code out of a loop that does not change is called loop", "invariant"),
        ("The technique of replacing a function call with its body is called", "inlining"),
        ("The technique of dividing a network into smaller parts is called", "segmentation"),
    ],
    "geographic_process_name": [
        ("The process by which fertile land becomes desert is called", "desertification"),
        ("The process by which cities grow at the expense of countryside is called", "urbanization"),
        ("The process by which a river bend is cut off to form a lake is called an oxbow", "cutoff"),
        ("The process by which a delta builds outward into the sea is called", "progradation"),
        ("The process by which land rises after ice sheets melt is called isostatic", "rebound"),
        ("The process by which a coastline is worn back by waves is called coastal", "erosion"),
        ("The process by which salt accumulates in irrigated soil is called", "salinization"),
        ("The process by which a lake fills with sediment and becomes land is called", "succession"),
    ],
    "ecological_process_name": [
        ("The gradual change of an ecological community over time is called", "succession"),
        ("The enrichment of water by nutrients causing algal blooms is called", "eutrophication"),
        ("The increase in a toxin's concentration up a food chain is called", "biomagnification"),
        ("The build-up of a substance within one organism over time is called", "bioaccumulation"),
        ("The relationship where both species benefit is called", "mutualism"),
        ("The relationship where one benefits and the other is unaffected is called", "commensalism"),
        ("The evolution of a species to fill many niches is called adaptive", "radiation"),
        ("The independent evolution of similar traits in unrelated species is called", "convergent"),
    ],
    # ---- structured counts (polygon_solid_count went 8/8) ------------------
    "music_interval_count": [
        ("The number of semitones in a perfect fifth is", "seven"),
        ("The number of semitones in a perfect fourth is", "five"),
        ("The number of semitones in a major third is", "four"),
        ("The number of semitones in a minor third is", "three"),
        ("The number of semitones in an octave is", "twelve"),
        ("The number of semitones in a major sixth is", "nine"),
        ("The number of semitones in a tritone is", "six"),
        ("The number of semitones in a major seventh is", "eleven"),
    ],
    "musical_ensemble_size": [
        ("The number of players in a string quartet is", "four"),
        ("The number of players in a piano trio is", "three"),
        ("The number of players in a woodwind quintet is", "five"),
        ("The number of strings on a standard violin is", "four"),
        ("The number of strings on a standard classical guitar is", "six"),
        ("The number of pedals on a standard grand piano is", "three"),
        ("The number of lines in a musical staff is", "five"),
        ("The number of valves on a standard trumpet is", "three"),
    ],
    "anatomical_count": [
        ("The number of bones in the adult human body is", "206"),
        ("The number of cervical vertebrae in humans is", "seven"),
        ("The number of pairs of ribs in humans is", "twelve"),
        ("The number of chambers in a fish heart is", "two"),
        ("The number of chambers in an amphibian heart is", "three"),
        ("The number of cranial nerves in humans is", "twelve"),
        ("The number of permanent teeth in an adult human is", "32"),
        ("The number of bones in the human ear ossicles is", "three"),
    ],
    "sporting_count": [
        ("The number of players on a rugby union team is", "fifteen"),
        ("The number of players on a rugby league team is", "thirteen"),
        ("The number of players on a water polo team in the water is", "seven"),
        ("The number of players on a netball team is", "seven"),
        ("The number of players on a volleyball team on court is", "six"),
        ("The number of players on a baseball team on the field is", "nine"),
        ("The number of players on an ice hockey team on the ice is", "six"),
        ("The number of players on a lacrosse team in the men's field game is", "ten"),
    ],
    "chemistry_count": [
        ("The number of electrons in a filled s subshell is", "two"),
        ("The number of electrons in a filled p subshell is", "six"),
        ("The number of electrons in a filled d subshell is", "ten"),
        ("The number of atoms in a molecule of ozone is", "three"),
        ("The number of covalent bonds carbon normally forms is", "four"),
        ("The number of protons in a helium nucleus is", "two"),
        ("The number of naturally occurring noble gases is", "six"),
        ("The number of electrons in a filled f subshell is", "fourteen"),
    ],
    # ---- work -> author / creator attribution (8/8 shape) -------------------
    "scientific_work_author": [
        ("The work On the Origin of Species was written by", "Darwin"),
        ("The work Principia Mathematica in physics was written by", "Newton"),
        ("The work The Interpretation of Dreams was written by", "Freud"),
        ("The work Silent Spring was written by", "Carson"),
        ("The work The Selfish Gene was written by", "Dawkins"),
        ("The work A Brief History of Time was written by", "Hawking"),
        ("The work De Revolutionibus was written by", "Copernicus"),
        ("The work Dialogue Concerning the Two Chief World Systems was written by", "Galileo"),
    ],
    "economic_work_author": [
        ("The work The Wealth of Nations was written by", "Smith"),
        ("The work Das Kapital was written by", "Marx"),
        ("The work The General Theory of Employment was written by", "Keynes"),
        ("The work The Road to Serfdom was written by", "Hayek"),
        ("The work Capital in the Twenty-First Century was written by", "Piketty"),
        ("The work An Essay on the Principle of Population was written by", "Malthus"),
        ("The work The Theory of Moral Sentiments was written by", "Smith"),
        ("The work Principles of Political Economy and Taxation was written by", "Ricardo"),
    ],
    "political_work_author": [
        ("The work The Prince was written by", "Machiavelli"),
        ("The work Democracy in America was written by", "Tocqueville"),
        ("The work On Liberty was written by", "Mill"),
        ("The work The Republic was written by", "Plato"),
        ("The work Politics, the classical treatise, was written by", "Aristotle"),
        ("The work Common Sense, the revolutionary pamphlet, was written by", "Paine"),
        ("The work The Spirit of the Laws was written by", "Montesquieu"),
        ("The work Two Treatises of Government was written by", "Locke"),
    ],
    "sociological_concept_author": [
        ("The concept of the Protestant work ethic in sociology is due to", "Weber"),
        ("The concept of anomie in sociology is due to", "Durkheim"),
        ("The concept of the looking-glass self is due to", "Cooley"),
        ("The concept of cultural capital is due to", "Bourdieu"),
        ("The concept of the panopticon as a social metaphor is due to", "Foucault"),
        ("The concept of the McDonaldization of society is due to", "Ritzer"),
        ("The concept of conspicuous consumption is due to", "Veblen"),
        ("The concept of the sociological imagination is due to", "Mills"),
    ],
    # ---- computed classification (language_word_order 7/8) -----------------
    "language_family_membership": [
        ("The language family that Finnish belongs to is", "Uralic"),
        ("The language family that Hungarian belongs to is", "Uralic"),
        ("The language family that Basque belongs to is a language", "isolate"),
        ("The language family that Swahili belongs to is", "Niger"),
        ("The language family that Turkish belongs to is", "Turkic"),
        ("The language family that Hebrew belongs to is", "Semitic"),
        ("The language family that Tamil belongs to is", "Dravidian"),
        ("The language family that Vietnamese belongs to is", "Austroasiatic"),
    ],
    "grammatical_case_function": [
        ("The grammatical case marking the direct object is the", "accusative"),
        ("The grammatical case marking possession is the", "genitive"),
        ("The grammatical case marking the indirect object is the", "dative"),
        ("The grammatical case marking the subject is the", "nominative"),
        ("The grammatical case marking the means by which something is done is the", "instrumental"),
        ("The grammatical case marking location is the", "locative"),
        ("The grammatical case used for direct address is the", "vocative"),
        ("The grammatical case marking motion away from is the", "ablative"),
    ],
    "logical_fallacy_name": [
        ("Attacking the person rather than the argument is the fallacy of ad", "hominem"),
        ("Misrepresenting an argument to attack it is the", "straw"),
        ("Assuming what you set out to prove is called begging the", "question"),
        ("Arguing that something is true because many believe it is an appeal to", "popularity"),
        ("Arguing a small step must lead to disaster is the slippery", "slope"),
        ("Presenting only two options when more exist is a false", "dilemma"),
        ("Concluding causation from sequence is post hoc ergo propter", "hoc"),
        ("Arguing something is true because an authority says so is an appeal to", "authority"),
    ],
    "statistical_measure_definition": [
        ("The measure of central tendency that is the middle value is the", "median"),
        ("The measure of central tendency that is the most frequent value is the", "mode"),
        ("The square root of the variance is the standard", "deviation"),
        ("The measure of the asymmetry of a distribution is called", "skewness"),
        ("The measure of the tailedness of a distribution is called", "kurtosis"),
        ("The difference between the upper and lower quartiles is the interquartile", "range"),
        ("The proportion of variance explained in a regression is called R", "squared"),
        ("The probability of rejecting a true null hypothesis is a Type", "I"),
    ],
    "geometry_term_definition": [
        ("A triangle with all sides of different lengths is called", "scalene"),
        ("A triangle with exactly two equal sides is called", "isosceles"),
        ("A quadrilateral with exactly one pair of parallel sides is a", "trapezoid"),
        ("A quadrilateral with all sides equal but angles not right is a", "rhombus"),
        ("An angle greater than ninety degrees but less than a straight angle is", "obtuse"),
        ("Two angles that sum to ninety degrees are called", "complementary"),
        ("Two angles that sum to one hundred eighty degrees are called", "supplementary"),
        ("A line that touches a circle at exactly one point is a", "tangent"),
    ],
    "medical_specialty_scope": [
        ("The medical specialty concerned with the heart is", "cardiology"),
        ("The medical specialty concerned with the kidneys is", "nephrology"),
        ("The medical specialty concerned with the nervous system is", "neurology"),
        ("The medical specialty concerned with hormones is", "endocrinology"),
        ("The medical specialty concerned with blood is", "hematology"),
        ("The medical specialty concerned with the skin is", "dermatology"),
        ("The medical specialty concerned with the digestive tract is", "gastroenterology"),
        ("The medical specialty concerned with joints and autoimmune disease is", "rheumatology"),
    ],
    "phobia_or_philia_object": [
        ("The fear of enclosed spaces is called", "claustrophobia"),
        ("The fear of open or public spaces is called", "agoraphobia"),
        ("The fear of heights is called", "acrophobia"),
        ("The fear of spiders is called", "arachnophobia"),
        ("The fear of water is called", "hydrophobia"),
        ("The fear of the number thirteen is called", "triskaidekaphobia"),
        ("The love of books is called", "bibliophilia"),
        ("The fear of foreigners or strangers is called", "xenophobia"),
    ],
    "study_of_field_name": [
        ("The scientific study of earthquakes is called", "seismology"),
        ("The scientific study of fungi is called", "mycology"),
        ("The scientific study of birds is called", "ornithology"),
        ("The scientific study of insects is called", "entomology"),
        ("The scientific study of fossils is called", "paleontology"),
        ("The scientific study of the origin of words is called", "etymology"),
        ("The scientific study of caves is called", "speleology"),
        ("The scientific study of soils is called", "pedology"),
    ],
    "measurement_scale_name": [
        ("The scale used to measure the hardness of minerals is the", "Mohs"),
        ("The scale used to measure the intensity of hurricanes is the", "Saffir"),
        ("The scale used to measure the intensity of tornadoes is the", "Fujita"),
        ("The scale used to measure wind force at sea is the", "Beaufort"),
        ("The scale used to measure the spiciness of chili peppers is the", "Scoville"),
        ("The scale used to measure acidity is the", "pH"),
        ("The scale used to measure the brightness of stars is called", "magnitude"),
        ("The scale used to measure sound intensity is the", "decibel"),
    ],
    "governance_term_definition": [
        ("Government by a small privileged class is called", "oligarchy"),
        ("Government by religious leaders is called", "theocracy"),
        ("Government by the people directly is called direct", "democracy"),
        ("Government by a single unelected ruler with absolute power is called", "autocracy"),
        ("Government by officials and administrators is called", "bureaucracy"),
        ("Government by the wealthy is called", "plutocracy"),
        ("A state with no government at all is called", "anarchy"),
        ("Government by those judged most able is called", "meritocracy"),
    ],
    "literary_device_name": [
        ("A comparison using like or as is called a", "simile"),
        ("Giving human qualities to non-human things is called", "personification"),
        ("Deliberate exaggeration for effect is called", "hyperbole"),
        ("The repetition of initial consonant sounds is called", "alliteration"),
        ("A contradiction in terms placed together is called an", "oxymoron"),
        ("A word that imitates the sound it denotes is called", "onomatopoeia"),
        ("Understatement by denying the opposite is called", "litotes"),
        ("A reference to another work or event is called an", "allusion"),
    ],
    "art_technique_name": [
        ("Painting on wet plaster so pigment binds to the wall is called", "fresco"),
        ("The technique of pasting materials onto a surface is called", "collage"),
        ("The strong contrast of light and dark in painting is called", "chiaroscuro"),
        ("The technique of painting with small dots of colour is called", "pointillism"),
        ("The technique of carving away material to reveal a form is called", "subtractive"),
        ("The blurring of outlines with subtle gradation is called", "sfumato"),
        ("Printing from an incised metal plate is called", "etching"),
        ("Painting with pigment suspended in egg yolk is called", "tempera"),
    ],
}


def rows_v5() -> list[dict]:
    return [{"prompt": f"Fact: {p}", "answer": f" {a}", "family": fam,
             "pool": "v5"}
            for fam, items in FAMILIES_V5.items() for p, a in items]


if __name__ == "__main__":
    r = rows_v5()
    print(f"v5: {len(FAMILIES_V5)} NEW canonical families, {len(r)} candidate "
          f"items ({len(r) / len(FAMILIES_V5):.1f} per family)")
