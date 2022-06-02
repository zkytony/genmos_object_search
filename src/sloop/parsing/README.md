# Parsing

Parse a natural language into graphical form that we can work with.

## Usage

**Added map_name argument to `parse()`**

Check out the Jupyter Notebook under "ipynb" directory. Here is a quick example.

The core functionality is wrapped in the `parse` function in `parser.py`.
The core API for the output of `parse` is in `spatial_lang.graph.spatial.SpatialGraph`.
Notice that `SpatialGraph` inherits the `spatial_lang.graph.graph.Graph` class.

### An example usage:
```python
kwfile = "../../data/language/fau_dor_keywords.json"
lang = "the bike is southwest of dorrance street, and is northwest"\
       "of providence hall, the bike is also directly south of the robot"
map_name = "dorrance"
sg = parse(lang, map_name, kwfile, spacy_model=spacy_model, verbose_level=2)
```
Let's break this down.
#### Keyword file
First, we supply `kwfile`, the path to a json keyword file that specifies keyword matching.
The keyword file for Faunce and Dorrance have been added to the repo under `data/language`.
This file looks like:
```
{
    "objects": {
        "GreenToyota": ["green toyota", "green car", "toyota", "car"],
        ...
    },
    "landmarks": {
        "WashingtonSt": ["Washington Street"],
        "DorranceSt": ["Dorrance Street"],
        ...
    },
    ...
```
Besides keywords for objects and landmarks this file also contains some specification
for other things, such as substituting certain phrases ("swaps"), and cosine similarity
thresholds ("_thresholds_").

#### Actual parsing
The parsing heavily relies on spaCy (Version 2.2.4). We are glad that spaCy is a popular
library with good community support.

When parsing the example sentence above, the `verbose_level=2` will cause the fucntion
to output many things. Among them, the line below indicates the original sentence has been
broken up (based on `nsubj` dependency label) into multiple sentences.
```
Subdoc 0: the bike is southwest of dorrance street, and is northwest of providence hall
...
```
You will also see matchings of objects and landmarks. The relational keyword matching is empty
because we only match relational keywords in very few circumstances (when it is followed by a `pobj`),
so as to influence the dependency tree structure to not drop the relational keyword. In most
cases, the relational keywords automatically gets picked up when we find paths between
entities.
```
Matched objects:
[('RedBike', bike)]
Matched Landmarks:
[('DorranceSt', dorrance street), ('ProvidenceHall', providence hall)]
Matched Relational Keywords:
[]
```
You will also see a cool visualization of the dependency tree (for the first sentence):
```
Printing dependency tree
           (ROOT) is
  _____________|________________________________
 |   |         |               |            (conj) is
 |   |         |               |                |
 |   |         |        (attr) southwest (attr) northwest
 |   |         |               |                |
 |   |  (nsubj) RedBike    (prep) of        (prep) of
 |   |         |               |                |
 ,  and       the          DorranceSt     ProvidenceHall 
```
Eventually, the function `parse` returns a `SpatialGraph` object `sg`, and
we can do the following to get the list of edges in the spatial graph
```python
sg.edges
# Outputs:
  {0: #0[<#1(RedBike)>is is northwest of<#2(ProvidenceHall)>],
   1: #1[<#1(RedBike)>is south of<#3(Robot)>],
   2: #2[<#1(RedBike)>is southwest of<#0(DorranceSt)>]}
```
We can also get a JSON representation of this spatial graph:
```python
sg.to_dict()
# Outputs:
  {'entities': ['DorranceSt', 'RedBike', 'ProvidenceHall', 'Robot'],
   'lang': 'the RedBike is southwest of DorranceSt, and is northwest of '
           'ProvidenceHall. the RedBike is also directly south of Robot',
   'relations': [('RedBike', 'ProvidenceHall', 'is is northwest of'),
                 ('RedBike', 'Robot', 'is south of'),
                 ('RedBike', 'DorranceSt', 'is southwest of')]}
```
A convenient function to save this JSON file is also available as `sg.to_file(path)`.

## Parse multiple sentences in the `.csv` data file
There is a script `parse_all.py` for this purpose. An example use:
```
python parse_all.py ../data/processed_data/amt_fau_dor_data.csv ../data/language/fau_dor_keywords.json output/
```
