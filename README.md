# Concept Trees of News Stories

This project extracts entities from one or multiple articles and detects relationships between them using [BabelNet](https://babelnet.org/) and Wikipedia. It then displays the entities and their relationships in a network.   
The different steps of the network are saved as html-files.

## Motivation 
In news articles, word choice used to refer to actors in the story tends to vary from neutral to biased. One way to induce bias is to transfer biased wording from one entity to a closely related entity.  In linguistics such relations are called meronymy, i.e., a semantic relation between a meronym denoting a part and a holonym denoting a whole (e.g., "Putin" - "Russia"), and metonymy, i.e., the substitution of the name of an attribute or adjunct for that of the thing meant (e.g., "the White House" - "the US government").  These relations may cause a news readers transferring framing one entity to another. For example, when reading about negative news about Putin, Russia is also perceived negatively due to Putin's role as a head of the country, i.e., a representative of the entire nation.

The goal of the project is to display relations between entities mentioned in an article. By extracting and labeling entities and their relations, we make a step towards identification of bias caused by metonymy/meronymy. A tree structure of representing entities and their relations within news articles enables news readers to: 1) have an overview of the news story participants, e.g., persons, organizations, countries, and their dependencies to each other, 2) assist in identification of bias that occurs via metonymy/meronymy. 

## Features
*Entity extraction using [spaCy](https://spacy.io/)*         
The entities from the articles are extracted using spaCy, which allows us to additionally calculate word-similarity.  

*Interactive networks*         
This project features interactive networks using [pyvis](https://pyvis.readthedocs.io/en/latest/): 

![network.png](https://github.com/RebeccaBraken/Concept-Trees/blob/main/network_example.png)

Additionally, different kinds of nodes are in different sizes. 
Edges are extracted from Wikipedia and Babelnet, newly added edges are colored in red:  

![new_edges.png](https://github.com/RebeccaBraken/Concept-Trees/blob/main/new_edges_example.png)   

*Similarity score*            
There is also a similarity score included, which is added to the network edges:   
![edges_labels.png](https://github.com/RebeccaBraken/Concept-Trees/blob/main/edge_labels_example.png)

*Filtering*          
There are different filtering methods included, which allow the User to filter by node or edge labels or for edges with a similarity score higher than a chosen threshold.
Here is an example of a network filtered by the node "United States":       
![filtered_network.png](https://github.com/RebeccaBraken/Concept-Trees/blob/main/nodefilter_example.png)


## Code examples    
Using spacy to extract named entities:
```python
nlp = spacy.load('en_core_web_lg')
nlp_article = nlp(articles)
```
nlp_article is a processed spacy-doc. We can use it to find entities and to filter them for their labels.

```python
for entity in nlp_article.ents:
    if entity.label_ not in unwanted_labels:
        nodes[entity] = {"label": ent.label_}
```

These entities can be cleaned and filtered and then added to the network.    
Edges are added if...     
...two entities from an article have a very high word-similarity (which is calculated using spaCy):
```python
if entity.similarity(other_entity) > sim_threshold:
    network.add_edge((entity, other_entity))
```
...two entities from an article appear in BabelNet and are connected there via an edge: 
```python
edges = api.get_outgoing_edges(id = entity_id)
if edges["target"].str.contains(other_entity_id).any():
    network.add_edge((entity, other_entity))
```
...one entity appears in the first _n_ sentences of the wikipedia-summary of the other entity:
```python
descr = wikipedia.summary(entity, sentences = n)
if other_entity in nlp(descr).ents:
    network.add_edge((entity, other_entity))
```
...two entities have an overlapping third entity in their wikipedia summary:
```python
overlapping_entities = [wiki_entity for entity in wiki_ents if wiki_entity in wiki_ents[entity]]
for wiki_entity in overlapping entities:
    network.add_edge((wiki_entity, entity))
    network.add_edge((wiki_entity, other_entity))
```
Furthermore, different kinds of filters can be applied to the resulting network. You can filter for node-/edge-labels and edges with a certain similarity score. 
Here is an example for filtering for edgelabels:
```python
filtered_network = nx.Graph([edge for edge in network.edges if any(label in edge for label in labels)])
```


## Installation        
     
### Requirements   
- python3.9 (64bit) 
- pip    

### .env
The project needs some parameters from a `.env` file.        
It needs to be stored in the same folder as the project.           
You will find an example `.env` file in the github repository.         
The `.env` needs to include:     

`BABELNET_KEY` A BabelNet api key. You can get one [here](https://babelnet.org/register).       
`SIMILARITY_THRESHOLD` All node-pairs with a higher wordsimilarity than this threshold will be connected via an edge in the network.     
`WIKIPEDIA_SENTENCES` The amount of sentences of the wikipedia-summary to use per entity when creating the network.     
`LOCATION_SIZE` The node-size of locations.     
`ORG_SIZE` The node-size of nodes which are Organizations / Companies.     
`OTHER_ENTITY_SIZE` The node-size of all other entities.     
`EDGES_WITHOUT_RELATIONSHIPTYPE` If set to False, BabelNet-edges with the unspecific label "Semantically related form" will not be added to the network.     
     
### Installation  
If you wish to install the packages into a virtual environment, you can do the following:
- Linux:
```
python3 -m venv /path_to_venv
source /path_to_venv/bin/activate
```
- Windows: 
```
python -m venv /path_to_venv
/path_to_venv/Scripts/activate
```

The packages you will need can be installed one by one:    
```
pip install pandas 		
pip install networkx	
pip install pyvis 		
pip install spacy 		
pip install wikipedia	
pip install py_babelnet
pip install python-dotenv
```
Or all at once:   
```
pip install pandas networkx pyvis spacy wikipedia py_babelnet python-dotenv
```

You additionally need to download a larger language package than the spacy standard package. This is needed for the word-similarity vectors:     
```
python -m spacy download en_core_web_lg
```

## Usage
To use this project, you need to run it using 
```
python concept_trees.py
```
The project reads data from .txt and .json-files.    
To give those files to the project, you can run the program with command line arguments such as 
```
python concept_trees.py text1.txt text2.json
```
The resulting networks will be saved to the project-folder so that the you can take a look at the different network generation steps.       
`wordsimilarity_network`       
The network with all edges with a word-similarity larger than the defined SIMILARITY_THRESHOLD.     
`babelnet_network`       
The network with the babelnet-edges on top of the wordsimilarity-network.     
`wikipedia_network_direct`         
The network with the direct wikipedia-edges on top of the babelnet_network.     
`wikipedia_network_indirect`       
The network with the indirect wikipedia-edges on top of wikipedia_network_direct.          
`weighted_network`       
Network with additional similarity score on the network edges.     
`filtered_edge_network`       
Network with an applied edge-filter. The other filtered networks are named analogical.     


## Roadmap
What I would have liked to add with more time:     
- Create dataframes of the edges for backup and optionally read dataframes to save BabelNet-keys.     
- Compare BabelNet-IDs of entity pairs to check if they are equal.     
- Optional directed networks.     
- Add multi-edges to include all of the different edge-labels.

## License
Licensed under the Apache License, Version 2.0 (the "License"); you may not use concept_trees except in compliance with the License. A copy of the License is included in the project, see the file [LICENSE](LICENSE).

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License
