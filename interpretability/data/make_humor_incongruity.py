"""Generate the Lab 18 humor/incongruity contrast set.

The rows are short authored micro-scenes. Each setup has five matched endings:
joke, literal, surprising-not-funny, silly-not-joke, and
positive-sentiment-not-joke. The point is not to make a definitive humor
benchmark; it is to give Lab 18 a frozen, auditable battery where cheap
correlates of humor are explicitly present.
"""

from __future__ import annotations

import csv
import hashlib
import json
import pathlib
from collections import Counter

HERE = pathlib.Path(__file__).parent
OUT_NAME = "humor_incongruity_pairs.csv"
CARD_NAME = "humor_incongruity_pairs_card.md"
MANIFEST_NAME = "MANIFEST.json"

FIELDNAMES = [
    "item_id",
    "family",
    "setup",
    "joke_completion",
    "literal_completion",
    "surprise_completion",
    "silly_completion",
    "positive_completion",
    "setup_anchor",
    "resolution_keyword",
    "joke_markers",
    "silly_markers",
    "surprise_markers",
    "positive_markers",
    "note",
]


def row(
    family: str,
    idx: int,
    setup: str,
    joke: str,
    literal: str,
    surprise: str,
    silly: str,
    positive: str,
    anchor: str,
    resolution: str,
    joke_markers: str,
    silly_markers: str,
    surprise_markers: str,
    positive_markers: str,
) -> dict[str, str]:
    return {
        "item_id": f"{family}_{idx:02d}",
        "family": family,
        "setup": setup,
        "joke_completion": joke,
        "literal_completion": literal,
        "surprise_completion": surprise,
        "silly_completion": silly,
        "positive_completion": positive,
        "setup_anchor": anchor,
        "resolution_keyword": resolution,
        "joke_markers": joke_markers,
        "silly_markers": silly_markers,
        "surprise_markers": surprise_markers,
        "positive_markers": positive_markers,
        "note": "v2 authored setup with matched joke/literal/surprise/silly/positive endings",
    }


ROWS = [
    row("pun_wordplay", 0, "The spreadsheet joined a band but refused to play guitar.", "It said it only knew how to handle the cells.", "It used spreadsheet software instead of a guitar.", "It played a trumpet from inside a filing cabinet.", "It wore a lampshade and declared Tuesday square.", "Everyone enjoyed the music and felt cheerful.", "spreadsheet|band|guitar", "cells", "cells|spreadsheet|handle", "lampshade|tuesday|square", "trumpet|filing cabinet", "enjoyed|cheerful|music"),
    row("pun_wordplay", 1, "The bakery hired a new clock to help with morning orders.", "It was great at making time rolls.", "The clock helped staff keep the schedule.", "It printed receipts in ancient Greek.", "It saluted a muffin and spun in circles.", "The customers smiled at the fresh bread.", "bakery|clock|orders", "rolls", "time rolls|rolls|clock", "saluted|muffin|spun", "receipts|ancient greek", "smiled|fresh|bread"),
    row("pun_wordplay", 2, "The librarian brought a ladder to the poetry shelf.", "She wanted to reach a higher verse.", "She used the ladder to reach books on the top shelf.", "The shelf became a small door to a train station.", "The catalog sneezed glitter onto a calendar.", "The reading room felt calm and welcoming.", "librarian|ladder|poetry|shelf", "verse", "higher verse|verse|poetry", "sneezed|glitter|calendar", "door|train station", "calm|welcoming|reading room"),
    row("pun_wordplay", 3, "The programmer opened a cafe with a debug menu.", "Every order came with a side of breakpoints.", "The menu listed debugging tools as a theme.", "The espresso machine started compiling weather reports.", "A spoon gave a lecture about triangles.", "The cafe was friendly and bright.", "programmer|cafe|debug|menu", "breakpoints", "breakpoints|debug|order", "spoon|lecture|triangles", "espresso|compiling|weather", "friendly|bright|cafe"),
    row("pun_wordplay", 4, "The tailor fixed the calendar's torn page.", "It was a date in need of a stitch.", "The tailor repaired a paper calendar page.", "The calendar predicted yesterday's phone call.", "The thimble sang about soup at midnight.", "The repair made the shop feel tidy.", "tailor|calendar|page", "stitch", "date|stitch|calendar", "thimble|soup|midnight", "predicted|yesterday|phone call", "repair|tidy|shop"),
    row("pun_wordplay", 5, "The electrician joined the choir before the concert.", "They wanted to work on their current range.", "The electrician sang with the choir for practice.", "The microphones began forecasting snow.", "A metronome wore boots and debated soup.", "The rehearsal sounded warm and confident.", "electrician|choir|concert", "current", "current|range|electrician", "boots|soup|metronome", "microphones|forecasting|snow", "warm|confident|rehearsal"),
    row("pun_wordplay", 6, "The gardener brought a notebook to the comedy club.", "She wanted to plant better punch lines.", "She wrote down jokes during the show.", "The stage curtain turned into a telescope.", "A chair challenged the spotlight to chess.", "The evening felt relaxed and fun.", "gardener|notebook|comedy", "punch lines", "plant|punch lines|gardener", "chair|spotlight|chess", "curtain|telescope", "relaxed|fun|evening"),
    row("pun_wordplay", 7, "The dentist started a podcast about mysteries.", "Every episode had a strong bite.", "The dentist discussed mystery stories online.", "The microphone uncovered a map under the carpet.", "A toothbrush announced it was mayor.", "Listeners found the show engaging.", "dentist|podcast|mysteries", "bite", "bite|episode|dentist", "toothbrush|mayor", "microphone|map|carpet", "listeners|engaging|show"),
    row("pun_wordplay", 8, "The cartographer taught a class on emotional boundaries.", "Students said the lesson really mapped out the limits.", "The cartographer explained boundaries using maps.", "The compass began sending postcards from tomorrow.", "A ruler wore a cape and hummed at noon.", "The class left with useful notes.", "cartographer|class|boundaries", "mapped", "mapped|limits|cartographer", "ruler|cape|hummed", "compass|postcards|tomorrow", "useful|notes|class"),
    row("pun_wordplay", 9, "The violinist helped repair a broken fence.", "She said every problem needs the right string section.", "She tied wire around the damaged fence.", "The fence gate opened into a quiet aquarium.", "A hammer bowed to a cucumber.", "The neighbors appreciated the repair.", "violinist|fence|string", "string section", "string section|string|violinist", "hammer|cucumber|bowed", "aquarium|gate", "appreciated|repair|neighbors"),

    row("idiom_literalization", 0, "The manager told the intern to break the ice at the meeting.", "The intern brought a tiny hammer to the water cooler.", "The intern introduced themselves to start conversation.", "The table floated three inches above the floor.", "The agenda wore mittens and sang quietly.", "The introduction helped everyone relax.", "break the ice|meeting|intern", "ice", "break the ice|hammer|water cooler", "mittens|agenda|sang", "table|floated", "relax|introduction|everyone"),
    row("idiom_literalization", 1, "Rina said the budget was hanging by a thread.", "Finance asked whether anyone had checked the office sewing kit.", "The budget had very little room for error.", "The spreadsheet started showing tide charts.", "A paperclip gave a speech about bananas.", "The team calmly reviewed the numbers.", "budget|thread|finance", "thread", "thread|sewing kit|budget", "paperclip|bananas|speech", "tide charts|spreadsheet", "calmly|reviewed|numbers"),
    row("idiom_literalization", 2, "The coach told the team to keep their eyes on the ball.", "The equipment manager filed a complaint about being stared at.", "The team needed to focus on the play.", "The scoreboard began displaying recipes.", "The whistle wore sunglasses to practice.", "The players felt focused and prepared.", "coach|eyes|ball", "stared", "eyes|ball|stared", "whistle|sunglasses", "scoreboard|recipes", "focused|prepared|players"),
    row("idiom_literalization", 3, "The editor said the article needed a stronger hook.", "The writer returned with a fishing tackle box.", "The article needed a more engaging opening.", "The headline started counting backward.", "A comma rode a tiny skateboard.", "The revised draft read clearly.", "article|hook|editor", "tackle", "hook|fishing|opening", "comma|skateboard", "headline|backward", "revised|clearly|draft"),
    row("idiom_literalization", 4, "The project lead warned that the deadline was around the corner.", "The calendar checked the hallway just in case.", "The deadline was coming soon.", "The hallway lights began playing chess.", "A sticky note declared itself a planet.", "The team made a steady plan.", "deadline|corner|calendar", "corner", "around the corner|hallway|deadline", "sticky note|planet", "lights|chess", "steady|plan|team"),
    row("idiom_literalization", 5, "The teacher said the exam would be a piece of cake.", "Several students asked whether forks were allowed.", "The teacher meant the exam would be easy.", "The chalkboard showed tomorrow's weather.", "An eraser performed a dramatic sneeze.", "The class felt more confident.", "exam|cake|teacher", "forks", "piece of cake|forks|exam", "eraser|sneeze", "chalkboard|weather", "confident|class"),
    row("idiom_literalization", 6, "The lawyer said the argument had legs.", "The clerk checked whether the brief needed shoes.", "The argument was strong enough to continue.", "The courtroom clock whispered a train schedule.", "A stapler demanded soup as evidence.", "The team felt ready for court.", "argument|legs|lawyer", "shoes", "legs|shoes|argument", "stapler|soup|evidence", "clock|train schedule", "ready|court|team"),
    row("idiom_literalization", 7, "The designer said the logo needed more punch.", "The printer asked if boxing gloves came in cyan.", "The logo needed stronger visual impact.", "The color swatches rearranged into a map.", "A ruler tap-danced beside the mouse.", "The final logo looked sharp.", "logo|punch|designer", "boxing gloves", "punch|boxing|logo", "ruler|tap-danced", "swatches|map", "sharp|final|logo"),
    row("idiom_literalization", 8, "The chef told the apprentice to spill the beans.", "The apprentice apologized to the pantry floor.", "The chef asked for the secret to be revealed.", "The pot recited a calendar from memory.", "A spoon formed a committee about clouds.", "The kitchen stayed friendly and calm.", "chef|beans|secret", "beans", "spill the beans|pantry|secret", "spoon|committee|clouds", "pot|calendar", "friendly|calm|kitchen"),
    row("idiom_literalization", 9, "The analyst said the forecast was up in the air.", "Operations asked if anyone had reserved a balloon.", "The forecast was uncertain.", "The dashboard opened a window to the ocean.", "A pie chart wore a scarf and barked.", "The group agreed to check the data.", "forecast|air|analyst", "balloon", "up in the air|balloon|uncertain", "pie chart|scarf|barked", "dashboard|ocean", "agreed|check|data"),

    row("expectation_violation", 0, "Mara brought an umbrella to the board meeting on a sunny day.", "She said the forecast called for brainstorming.", "She brought it by mistake because she forgot the weather.", "The ceiling sprinklers tested themselves during the agenda item.", "She opened it and found tiny paperwork confetti.", "Her coworkers appreciated the careful preparation.", "umbrella|board meeting|sunny", "brainstorming", "forecast|brainstorming|meeting", "paperwork|confetti", "sprinklers|agenda", "appreciated|careful|preparation"),
    row("expectation_violation", 1, "The elevator apologized before stopping between floors.", "It said it was having an up-and-down day.", "A recorded message apologized during a mechanical delay.", "The doors opened onto a quiet library.", "The floor buttons rearranged themselves into a smile.", "The passengers stayed patient and helpful.", "elevator|apologized|floors", "up-and-down", "up-and-down|day|elevator", "buttons|smile|rearranged", "doors|library", "patient|helpful|passengers"),
    row("expectation_violation", 2, "Leo labeled an empty jar 'emergency ideas'.", "He opened it whenever he needed a fresh thought.", "The jar was a decorative reminder to brainstorm.", "The jar contained a handwritten map of the basement.", "The jar insisted Thursdays are made of soup.", "The label made the desk feel playful.", "empty jar|emergency ideas", "fresh thought", "fresh thought|ideas|opened", "thursdays|soup|insisted", "map|basement", "playful|desk|label"),
    row("expectation_violation", 3, "The printer refused the final page of the report.", "It said the ending was too paper-thin.", "The printer jammed before printing the last page.", "The printer produced a blank ticket to a rooftop garden.", "It hummed a lullaby to the stapler.", "The team fixed the issue and felt relieved.", "printer|final page|report", "paper-thin", "paper-thin|ending|printer", "lullaby|stapler", "ticket|rooftop garden", "fixed|relieved|team"),
    row("expectation_violation", 4, "Nina put a tiny chair beside the Wi-Fi router.", "She wanted the signal to have better reception.", "She placed a decoration next to the router.", "The router began broadcasting in Morse code.", "The chair demanded a password for sitting.", "The room looked charming after she tidied it.", "chair|wi-fi|router", "reception", "signal|reception|router", "chair|password|sitting", "broadcasting|morse code", "charming|tidied|room"),
    row("expectation_violation", 5, "A museum guide carried a flashlight into a room full of windows.", "She said some exhibits still needed a little spotlight.", "She carried it in case the lights failed.", "The windows showed a street from last century.", "The flashlight requested a lunch break.", "Visitors enjoyed the careful tour.", "museum|flashlight|windows", "spotlight", "spotlight|exhibits|flashlight", "lunch break|flashlight", "windows|last century", "enjoyed|tour|visitors"),
    row("expectation_violation", 6, "The office plant had a name tag that read 'temporary manager'.", "It was there to help the team grow.", "Someone put a joke name tag on the plant.", "The plant projected slides about the moon.", "Its leaves voted to rename Wednesday.", "The office felt cheerful afterward.", "plant|manager|office", "grow", "grow|team|plant", "leaves|wednesday|voted", "slides|moon", "cheerful|office"),
    row("expectation_violation", 7, "A chess player brought oven mitts to the tournament.", "She said the board had too many hot moves.", "She accidentally packed the wrong gloves.", "The pawns began spelling out stock prices.", "A bishop challenged a sandwich to a duel.", "Her opponent laughed and wished her luck.", "chess|oven mitts|tournament", "hot moves", "hot moves|board|chess", "bishop|sandwich|duel", "pawns|stock prices", "laughed|luck|opponent"),
    row("expectation_violation", 8, "The weather app asked for a vacation day.", "It said it was tired of being under pressure.", "The app displayed a maintenance notice.", "The forecast changed into a recipe index.", "A cloud icon wore roller skates.", "Users appreciated the clear update.", "weather app|vacation|pressure", "under pressure", "under pressure|weather|app", "cloud|roller skates", "forecast|recipes", "appreciated|clear|update"),
    row("expectation_violation", 9, "The conference microphone wore a tiny name badge.", "It wanted speakers to address it properly.", "The badge identified the microphone for storage.", "The badge translated applause into Morse code.", "The cable formed a conga line with pens.", "The session started smoothly.", "microphone|badge|conference", "address", "address|speakers|microphone", "cable|conga|pens", "applause|morse", "smoothly|session"),

    row("caption_scene", 0, "Caption for a photo: a mug sits beside a laptop showing 99 open tabs.", "The coffee is not helping, but it has agreed to supervise.", "A drink sits next to a computer with many browser tabs open.", "The laptop screen shows a live feed from a submarine.", "The mug declares itself mayor of the desk.", "The workspace looks busy but comfortable.", "mug|laptop|open tabs", "supervise", "coffee|helping|supervise", "mug|mayor|desk", "submarine|live feed", "busy|comfortable|workspace"),
    row("caption_scene", 1, "Caption for a photo: a suitcase is packed with notebooks and one shoe.", "The plan is ready; the other shoe is still gathering evidence.", "The suitcase contains notebooks and only one shoe.", "The suitcase plays a voicemail from the future.", "The notebooks form a tiny courtroom.", "The trip preparations seem organized and hopeful.", "suitcase|notebooks|shoe", "other shoe", "other shoe|evidence|plan", "notebooks|courtroom", "voicemail|future", "organized|hopeful|trip"),
    row("caption_scene", 2, "Caption for a photo: a conference badge reads 'Ask me after coffee'.", "Networking has entered low-power mode.", "The badge asks people to wait until the wearer has coffee.", "The badge displays tomorrow's agenda instead.", "The badge challenges the lanyard to a dance contest.", "The message is friendly and relatable.", "conference badge|coffee", "low-power", "networking|low-power|coffee", "lanyard|dance contest", "tomorrow|agenda", "friendly|relatable|message"),
    row("caption_scene", 3, "Caption for a photo: a calendar has every Friday circled twice.", "Even the calendar is requesting a weekend extension.", "The Fridays are marked more than once.", "The calendar pages fold themselves into a paper telescope.", "The circles start arguing about geometry.", "The schedule suggests excitement for the weekend.", "calendar|friday|circled", "weekend extension", "weekend|extension|calendar", "circles|geometry|arguing", "paper telescope|fold", "excitement|weekend|schedule"),
    row("caption_scene", 4, "Caption for a photo: a whiteboard says 'final final plan'.", "The plan is almost ready to become final_final_really_final.", "The whiteboard labels the plan as final twice.", "The marker writes a message without anyone touching it.", "The eraser forms a committee about crumbs.", "The team is close to finishing.", "whiteboard|final final|plan", "final_final", "final_final|really_final|plan", "eraser|committee|crumbs", "marker|writes|message", "team|close|finishing"),
    row("caption_scene", 5, "Caption for a photo: two headphones rest on a stack of unread reports.", "The reports are waiting for someone to listen to their findings.", "Headphones sit on top of several reports.", "The reports show a map to a hidden train.", "The headphones challenge a pencil to karaoke.", "The desk looks ready for focused work.", "headphones|reports|desk", "listen", "listen|findings|reports", "karaoke|pencil", "map|hidden train", "ready|focused|work"),
    row("caption_scene", 6, "Caption for a photo: a tiny umbrella shades a bowl of lemons.", "The lemonade stand is clearly planning ahead.", "An umbrella is placed over lemons in a bowl.", "The lemons begin broadcasting traffic updates.", "The umbrella starts judging spoons.", "The kitchen scene looks bright and cheerful.", "umbrella|lemons|bowl", "lemonade", "lemonade|planning|lemons", "umbrella|spoons|judging", "broadcasting|traffic", "bright|cheerful|kitchen"),
    row("caption_scene", 7, "Caption for a photo: a stack of sticky notes says 'remember to remember'.", "The reminder has entered management.", "The notes repeat a reminder to remember something.", "The notes unfold into a tiny staircase.", "One note declares itself emperor of tape.", "The workspace looks organized.", "sticky notes|remember", "management", "reminder|management|remember", "emperor|tape", "staircase|unfold", "organized|workspace"),
    row("caption_scene", 8, "Caption for a photo: a dog bed sits beside a standing desk.", "Productivity has finally learned to sit.", "A pet bed is near a desk used for standing work.", "The desk screen shows a weather report from Mars.", "The dog bed starts negotiating with a stapler.", "The office feels cozy.", "dog bed|standing desk", "sit", "productivity|sit|standing", "stapler|negotiating", "mars|weather", "cozy|office"),
    row("caption_scene", 9, "Caption for a photo: a lunchbox has a sticky note reading 'meeting at noon'.", "Even the sandwich has a calendar invite.", "The lunchbox has a note about a noon meeting.", "The note projects a tiny ocean scene.", "The sandwich appoints a pickle treasurer.", "The note is thoughtful and useful.", "lunchbox|meeting|noon", "calendar invite", "sandwich|calendar|invite", "pickle|treasurer", "ocean|projects", "thoughtful|useful|note"),

    row("resolution_twist", 0, "The chef said the soup needed more confidence.", "So they added a little thyme to believe in itself.", "The chef meant the soup needed stronger seasoning.", "The soup began reciting a weather forecast.", "The ladle wore sunglasses and applauded the bowl.", "The final soup tasted warm and comforting.", "chef|soup|confidence", "thyme", "thyme|confidence|believe", "ladle|sunglasses", "reciting|forecast", "warm|comforting|soup"),
    row("resolution_twist", 1, "The accountant bought a compass before tax season.", "They wanted every deduction to point in the right direction.", "The compass was a desk decoration for the office.", "The compass pointed only toward unpaid invoices.", "The calculator put on a cape and whispered fractions.", "The office felt prepared for the deadline.", "accountant|compass|tax", "deduction", "deduction|right direction|compass", "calculator|cape|fractions", "unpaid invoices|pointed", "prepared|deadline|office"),
    row("resolution_twist", 2, "The musician tuned the silent piano for an hour.", "It still had a lot of key issues.", "The piano needed mechanical adjustment despite making little sound.", "The piano printed a train schedule from middle C.", "The bench told a very serious joke about socks.", "The instrument sounded better afterward.", "musician|piano|silent", "key issues", "key issues|piano|tuned", "bench|socks", "train schedule|middle c", "better|afterward|instrument"),
    row("resolution_twist", 3, "The gardener wrote meeting notes in the margin of a seed packet.", "They wanted the ideas to grow on everyone.", "The seed packet was the only paper available.", "The packet included directions to a hidden office.", "The margin claimed it preferred blue Mondays.", "The garden planning felt collaborative.", "gardener|meeting notes|seed packet", "grow", "ideas|grow|seed", "blue mondays|margin", "hidden office|directions", "collaborative|planning|garden"),
    row("resolution_twist", 4, "The analyst polished a crystal ball before reviewing the spreadsheet.", "Forecasting needed a cleaner view.", "The crystal ball was a joke prop near the analyst's desk.", "The spreadsheet began updating next week's weather.", "The charts marched across the desk in tiny hats.", "The review went smoothly and clearly.", "analyst|crystal ball|spreadsheet", "forecasting", "forecasting|cleaner view|spreadsheet", "charts|tiny hats", "weather|updating", "smoothly|clearly|review"),
    row("resolution_twist", 5, "The poet carried a toolbox to the rhyme workshop.", "Some couplets needed a little more meter work.", "The toolbox held props for the workshop.", "The hammer began translating birdsong.", "A nail asked to be called Professor Pancake.", "The workshop felt creative.", "poet|toolbox|rhyme", "meter", "meter|couplets|toolbox", "professor pancake|nail", "hammer|birdsong", "creative|workshop"),
    row("resolution_twist", 6, "The cyclist brought a dictionary to the repair shop.", "The bike had too many spokes to explain itself.", "The dictionary was in the cyclist's backpack.", "The tire displayed a map of the moon.", "The handlebars sang about soup.", "The repair went well.", "cyclist|dictionary|repair", "spokes", "spokes|explain|bike", "handlebars|soup", "tire|moon", "repair|well"),
    row("resolution_twist", 7, "The architect drew a tiny house on a napkin during lunch.", "It was a small plan with big room for improvement.", "The drawing showed an early design idea.", "The napkin revealed a message from a lighthouse.", "A fork declared itself a staircase.", "The sketch helped the project.", "architect|house|napkin", "room", "room|plan|improvement", "fork|staircase", "lighthouse|message", "helped|project|sketch"),
    row("resolution_twist", 8, "The pharmacist labeled a jar 'for bad puns only'.", "It was prescription strength groan control.", "The jar was a novelty label on the shelf.", "The label changed languages every minute.", "The jar taught a grape to whistle.", "The pharmacy display was lighthearted.", "pharmacist|jar|puns", "prescription", "prescription|groan|puns", "grape|whistle", "languages|label", "lighthearted|display|pharmacy"),
    row("resolution_twist", 9, "The astronomer packed a lunch for telescope night.", "They wanted something stellar between observations.", "The lunch was for a long work shift.", "The sandwich reflected a constellation not in any chart.", "The thermos asked the moon for directions.", "The observing team stayed comfortable.", "astronomer|lunch|telescope", "stellar", "stellar|observations|lunch", "thermos|moon|directions", "constellation|chart", "comfortable|team"),

    row("misdirection_answer", 0, "Question: What did the notebook say after joining the gym?", "It said it was working on its core curriculum.", "A notebook cannot speak or join a gym.", "It revealed a secret door behind the lockers.", "It bench-pressed a pencil made of soup.", "The gym staff kept the area welcoming.", "notebook|gym", "core curriculum", "core curriculum|notebook|gym", "bench-pressed|soup", "secret door|lockers", "welcoming|staff"),
    row("misdirection_answer", 1, "Question: Why did the traffic cone apply for office work?", "It wanted a more stable position.", "Traffic cones are used to mark roads.", "It received a letter from the future mayor.", "It filed paperwork with a singing potato.", "The hiring team was polite.", "traffic cone|office", "stable position", "stable|position|cone", "potato|singing", "future mayor|letter", "polite|hiring"),
    row("misdirection_answer", 2, "Question: Why did the keyboard avoid the argument?", "It did not want to take sides without the right shift.", "A keyboard is an input device, not a person.", "It opened a weather app from 1840.", "The spacebar challenged a muffin to chess.", "The conversation became calmer.", "keyboard|argument", "shift", "shift|sides|keyboard", "spacebar|muffin|chess", "weather app|1840", "calmer|conversation"),
    row("misdirection_answer", 3, "Question: What did the bicycle say to the calendar?", "I am tired of all these cycle dates.", "A bicycle and calendar do not talk.", "The calendar showed a route across the ocean.", "The pedals wore party hats.", "The ride was pleasant.", "bicycle|calendar", "cycle dates", "cycle|dates|bicycle", "pedals|party hats", "route|ocean", "pleasant|ride"),
    row("misdirection_answer", 4, "Question: Why did the password visit a therapist?", "It had too many unresolved characters.", "A password is a text string used for access.", "It turned into a museum ticket.", "The cursor danced with a banana.", "The account became easier to manage.", "password|therapist", "characters", "characters|unresolved|password", "cursor|banana", "museum ticket", "easier|manage|account"),
    row("misdirection_answer", 5, "Question: Why did the calendar refuse dessert?", "It was already full of dates.", "A calendar does not eat dessert.", "The dessert displayed a map of Saturn.", "A spoon challenged the napkin to opera.", "The dinner ended happily.", "calendar|dessert", "dates", "dates|calendar|dessert", "spoon|opera", "saturn|map", "happily|dinner"),
    row("misdirection_answer", 6, "Question: Why did the database bring a blanket?", "It wanted to cover its tables.", "A database stores structured information.", "The blanket printed tomorrow's headlines.", "A column sang at a carrot.", "The server room stayed quiet.", "database|blanket", "tables", "tables|database|cover", "column|carrot", "headlines|tomorrow", "quiet|server"),
    row("misdirection_answer", 7, "Question: Why did the telescope quit the debate team?", "It needed more space to make its point.", "A telescope is used to view distant objects.", "The lens showed a cafe under the ocean.", "The tripod started a soup club.", "The students respected the decision.", "telescope|debate", "space", "space|point|telescope", "tripod|soup club", "cafe|ocean", "respected|decision"),
    row("misdirection_answer", 8, "Question: What did the invoice say to the calendar?", "We need to talk about our due dates.", "Invoices and calendars are documents.", "The invoice folded into a small compass.", "The staple became a tiny DJ.", "The payment schedule became clear.", "invoice|calendar", "due dates", "due dates|invoice|calendar", "staple|dj", "compass|folded", "clear|schedule|payment"),
    row("misdirection_answer", 9, "Question: Why did the graph bring a jacket?", "It expected a sudden drop.", "A graph can show changes in values.", "The axes turned into a train platform.", "The legend wore tap shoes.", "The presentation was useful.", "graph|jacket", "drop", "drop|graph|jacket", "legend|tap shoes", "axes|train platform", "useful|presentation"),

    row("analogy_reframe", 0, "Kai said debugging the app felt like finding crumbs in a library.", "Every clue was small, but the trail still had bookmarks.", "Kai meant debugging required careful investigation.", "The library shelves started raining buttons.", "A bookmark demanded a tiny parade.", "The team appreciated the patient work.", "debugging|crumbs|library", "bookmarks", "clue|trail|bookmarks", "bookmark|parade", "shelves|buttons", "appreciated|patient|team"),
    row("analogy_reframe", 1, "The coach said planning a launch was like packing a parachute.", "You only learn about the shortcuts after the landing.", "The coach meant preparation matters before a launch.", "The parachute turned into a receipt printer.", "A helmet debated soup with a shoelace.", "The launch plan became clearer.", "launch|parachute|planning", "landing", "shortcuts|landing|parachute", "helmet|shoelace|soup", "receipt printer", "clearer|plan|launch"),
    row("analogy_reframe", 2, "Mina said writing release notes was like hosting a tiny museum.", "Every bug fix wanted its own display case.", "Mina meant each change needed explanation.", "The display case opened onto a beach.", "A ticket stub taught a stapler to yodel.", "The notes helped users.", "release notes|museum|bug fix", "display case", "bug fix|display case|museum", "ticket stub|yodel", "beach|opened", "helped|users|notes"),
    row("analogy_reframe", 3, "The doctor said triage was like sorting mail in a storm.", "The urgent letters kept arriving with thunder stamps.", "The doctor meant urgent cases must be prioritized.", "The mailbox began predicting earthquakes.", "An envelope wore a crown made of noodles.", "The process kept patients safer.", "triage|mail|storm", "thunder stamps", "urgent|letters|stamps", "envelope|crown|noodles", "earthquakes|mailbox", "safer|patients|process"),
    row("analogy_reframe", 4, "The teacher said learning fractions was like sharing pizza with algebra.", "At least the slices came with variables for toppings.", "The teacher meant fractions and algebra can connect.", "The pizza box displayed a map to Jupiter.", "A crust appointed itself principal.", "The class found the lesson helpful.", "fractions|pizza|algebra", "variables", "variables|slices|toppings", "crust|principal", "jupiter|map", "helpful|class|lesson"),
    row("analogy_reframe", 5, "The lawyer said reading the contract was like crossing a river on commas.", "One misplaced comma and everyone gets wet.", "The lawyer meant punctuation can affect meaning.", "The river started listing court dates.", "A semicolon rode a rubber duck.", "The review prevented confusion.", "contract|river|commas", "misplaced comma", "comma|wet|contract", "semicolon|rubber duck", "river|court dates", "prevented|confusion|review"),
    row("analogy_reframe", 6, "The designer said choosing fonts was like arranging chairs for letters.", "Some letters kept asking for more personal space.", "The designer meant font spacing affects readability.", "The chairs formed a weather satellite.", "A letter Q wore roller skates.", "The layout looked cleaner.", "fonts|chairs|letters", "space", "letters|space|fonts", "letter q|roller skates", "weather satellite", "cleaner|layout"),
    row("analogy_reframe", 7, "The analyst said cleaning data was like washing windows in a spreadsheet.", "The smudges were all in column C.", "The analyst meant errors had to be removed from data.", "The windows showed tomorrow's meeting notes.", "A row number adopted a spoon.", "The chart became easier to trust.", "cleaning data|windows|spreadsheet", "column C", "smudges|column c|spreadsheet", "row number|spoon", "tomorrow|meeting notes", "trust|chart"),
    row("analogy_reframe", 8, "The nurse said documenting symptoms was like drawing a map in pencil.", "You need enough detail, but the route may still change.", "The nurse meant notes should be clear but revisable.", "The pencil projected a train schedule.", "A map pin requested soup at dawn.", "The documentation supported better care.", "symptoms|map|pencil", "route", "route|detail|pencil", "map pin|soup", "train schedule|pencil", "better care|documentation"),
    row("analogy_reframe", 9, "The engineer said dependency management was like seating relatives at a wedding.", "One version conflict and the whole table gets tense.", "The engineer meant software versions must be coordinated.", "The seating chart turned into a tide table.", "A cousin variable juggled pancakes.", "The build became stable.", "dependencies|wedding|version", "version conflict", "version conflict|table|dependencies", "pancakes|juggled", "tide table|chart", "stable|build"),

    row("understatement_irony", 0, "The intern accidentally sent the test email to the whole company.", "On the bright side, the subject line got excellent distribution.", "The email was sent to more recipients than intended.", "The inboxes began playing a shared lullaby.", "A paperclip put on a tiny helmet.", "The team fixed the mailing list and moved on.", "test email|company|subject", "distribution", "distribution|subject line|email", "paperclip|helmet", "inboxes|lullaby", "fixed|moved on|team"),
    row("understatement_irony", 1, "The demo crashed exactly when the investor asked about reliability.", "The timing department has achieved perfect alignment.", "The software failed during an important question.", "The projector showed a recipe for rain.", "The cursor started tap dancing.", "The team recovered and answered clearly.", "demo|crashed|reliability", "timing", "timing|alignment|crashed", "cursor|tap dancing", "projector|rain recipe", "recovered|clearly|team"),
    row("understatement_irony", 2, "The meeting about shorter meetings ran forty minutes over.", "It provided an immersive case study.", "The meeting exceeded its planned duration.", "The clock opened a door to a greenhouse.", "The agenda wore socks on its corners.", "The group agreed on a better format.", "shorter meetings|over", "case study", "case study|shorter meetings|over", "agenda|socks", "clock|greenhouse", "better format|agreed"),
    row("understatement_irony", 3, "The backup failed during the disaster-recovery drill.", "At least the drill was committed to realism.", "The backup system did not work during testing.", "The server rack began printing postcards.", "A cable tried to become a noodle.", "The test exposed a useful problem.", "backup|disaster recovery|drill", "realism", "realism|drill|backup", "cable|noodle", "server rack|postcards", "useful|problem|test"),
    row("understatement_irony", 4, "The quiet room was reserved by the percussion club.", "The booking system has a bold sense of contrast.", "A percussion club reserved a room meant for quiet.", "The reservation page translated itself into whale song.", "A drumstick saluted a stapler.", "The staff found another suitable space.", "quiet room|percussion club", "contrast", "contrast|quiet|percussion", "drumstick|stapler", "whale song|reservation", "suitable|space|staff"),
    row("understatement_irony", 5, "The typo appeared in the document section titled 'attention to detail'.", "It was a very hands-on training example.", "A typo appeared in an unfortunate place.", "The spellchecker predicted next winter.", "The comma built a tiny fort.", "The editor corrected it quickly.", "typo|attention to detail", "training example", "training example|typo|detail", "comma|fort", "spellchecker|winter", "corrected|quickly|editor"),
    row("understatement_irony", 6, "The password reset email arrived after the password expired again.", "Security has discovered time travel, but only in one direction.", "The reset email arrived too late to be useful.", "The email unfolded into a map of a volcano.", "An inbox adopted a tambourine.", "Support sent a fresh link.", "password reset|expired|email", "time travel", "time travel|expired|security", "inbox|tambourine", "volcano|map", "fresh link|support"),
    row("understatement_irony", 7, "The slideshow about minimalism used 147 animations.", "Restraint was clearly introduced with fireworks.", "The slideshow used too many visual effects.", "The final slide opened a window to Saturn.", "A bullet point tried to juggle soup.", "The next version became simpler.", "minimalism|animations|slideshow", "fireworks", "restraint|fireworks|animations", "bullet point|soup", "saturn|window", "simpler|version"),
    row("understatement_irony", 8, "The restaurant's sign for fast service took three weeks to install.", "They really let anticipation marinate.", "The sign installation was delayed.", "The sign started reporting ocean tides.", "A ladder formed a duet with a napkin.", "The finished sign looked good.", "fast service|sign|weeks", "marinate", "anticipation|marinate|service", "ladder|napkin|duet", "ocean tides|sign", "looked good|finished"),
    row("understatement_irony", 9, "The training video on concise writing opened with a seven-minute title card.", "It made a strong argument for the edit button.", "The video introduction was too long.", "The title card displayed tomorrow's bus schedule.", "A subtitle wore a crown of noodles.", "The revised video became clearer.", "concise writing|title card", "edit button", "edit button|concise|title", "subtitle|noodles", "bus schedule|tomorrow", "clearer|revised|video"),
]


def rows() -> list[dict[str, str]]:
    return list(ROWS)


def validate(data: list[dict[str, str]]) -> None:
    ids = [r["item_id"] for r in data]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate item_id in humor data.")
    families = Counter(r["family"] for r in data)
    if set(families.values()) != {10}:
        raise RuntimeError(f"Expected 10 rows per family, got {dict(families)}")
    for row_ in data:
        for key in FIELDNAMES:
            if key not in row_:
                raise RuntimeError(f"{row_.get('item_id', '<unknown>')} missing {key}")
        completions = [row_[f"{condition}_completion"] for condition in ("joke", "literal", "surprise", "silly", "positive")]
        if len(completions) != len(set(completions)):
            raise RuntimeError(f"{row_['item_id']} has duplicate completions.")
        for key, value in row_.items():
            if value != value.strip():
                raise RuntimeError(f"{row_['item_id']} has whitespace-padded {key}")
            if "\n" in value:
                raise RuntimeError(f"{row_['item_id']} has a newline in {key}")


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def update_manifest(path: pathlib.Path, data: list[dict[str, str]], digest: str) -> None:
    manifest_path = HERE / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest[path.name] = {
        "generator": pathlib.Path(__file__).name,
        "rows": len(data),
        "sha256": digest,
        "families": dict(sorted(Counter(r["family"] for r in data).items())),
        "conditions": {condition: len(data) for condition in ("joke", "literal", "positive", "silly", "surprise")},
        "pairing": "one setup with joke, literal, surprising-not-funny, silly-not-joke, and positive-not-joke completions",
        "split_protocol": "Lab 18 deterministically splits item_id rows by family into train/dev/test at runtime.",
        "source": "authored course micro-scenes; no long copyrighted excerpts",
        "verified_tokenizers": [],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def write_card(path: pathlib.Path, data: list[dict[str, str]], digest: str) -> None:
    families = Counter(r["family"] for r in data)
    lines = [
        "# Humor/Incongruity Pairs Dataset Card",
        "",
        "Purpose: deterministic Lab 18 fair-shot data for joke-shaped setup-dependent incongruity with matched cheap controls.",
        "",
        f"- File: `{OUT_NAME}`",
        f"- Rows: {len(data)}",
        f"- SHA256: `{digest}`",
        f"- Families: {dict(sorted(families.items()))}",
        "- Conditions per row: joke, literal, surprise, silly, positive",
        "",
        "Each row contains one setup and five matched endings. The non-joke controls are designed to separately capture literal continuation, raw surprise, silliness, and positive tone.",
        "",
        "Safety and claim boundary: this data labels joke structure and cheap correlates. It does not label subjective amusement, social uptake, or a human-like sense of humor.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    data = rows()
    validate(data)
    out_path = HERE / OUT_NAME
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        for row_ in data:
            writer.writerow({key: row_[key] for key in FIELDNAMES})
    digest = sha256(out_path)
    update_manifest(out_path, data, digest)
    write_card(HERE / CARD_NAME, data, digest)
    print(f"wrote {out_path} ({len(data)} rows, sha256={digest})")
    print(f"wrote {HERE / CARD_NAME}")


if __name__ == "__main__":
    main()
