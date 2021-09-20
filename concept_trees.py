import networkx as nx
from pyvis.network import Network

import re

import spacy
import wikipedia

from py_babelnet.calls import BabelnetAPI
import py_babelnet

import pandas as pd

import os
from dotenv import load_dotenv

import json
import sys

import time
import socket
from json.decoder import JSONDecodeError


#initialize the api-key
def init_bn_api():
    
    load_dotenv()
    key = os.getenv("BABELNET_KEY")
    if key is None:
        raise ValueError("No BABELNET_KEY found in .env-file. Please check your .env and try again.")
    api = BabelnetAPI(key)
    
    return api


#initialize the values needed for the program - read all neccessary data from a .env-file
def init_values():

    load_dotenv()
    similarity_threshold = float(os.getenv("SIMILARITY_THRESHOLD"))
    wikipedia_sentences = int(os.getenv("WIKIPEDIA_SENTENCES"))
    location_size = int(os.getenv("LOCATION_SIZE"))
    org_size = int(os.getenv("ORG_SIZE"))
    entity_size = int(os.getenv("OTHER_ENTITY_SIZE"))
    unlabelled_edges = os.getenv("EDGES_WITHOUT_RELATIONSHIPTYPE", 'False').lower() in ('true', '1', 't')

    if similarity_threshold is None:
        raise ValueError("No SIMILARITY_THRESHOLD found in .env-file. Please check your .env-file and try again.")
    if wikipedia_sentences is None:
        raise ValueError("No WIKIPEDIA_SENTENCES found in .env-file. Please check your .env-file and try again.")
    if location_size is None:
        raise ValueError("No LOCATION_SIZE found in .env-file. Please check your .env-file and try again.")
    if org_size is None:
        raise ValueError("No ORG_SIZE found in .env-file. Please check your .env-file and try again.")
    if entity_size is None:
        raise ValueError("No OTHER_ENTITY_SIZE found in .env-file. Please check your .env-file and try again.")
    if unlabelled_edges is None:
        raise ValueError("No EDGES_WITHOUT_RELATIONSHIPTYPE found in .env-file. Please check your .env-file and try again.")

    return similarity_threshold, wikipedia_sentences, location_size, org_size, entity_size, unlabelled_edges


#read the articles from the given files and return them in a string
def read_articles(filenames: list):
    
    all_articles = ""
    for filename in filenames:
        # try to read article from a json file
        if filename.endswith(".json"):
            try:
                with open(filename) as f:
                    data = json.load(f)
            except IOError:
                print("Error: File {} does not appear to exist.".format(filename))
            except JSONDecodeError :
                print("Error: File {} is not in JSON-format.".format(filename))
            else:
                try:
                    article = data["text"]
                except AttributeError:
                    print("No text object found in json-file {}".format(filename))
                else:
                    all_articles += article
        # try to read article from a .txt-file
        elif filename.endswith(".txt"): 
            try:
                with open(filename) as f:
                    data = f.read()
            except IOError:
                print("Error: File {} does not appear to exist.".format(filename))
            else:
                all_articles += data
    
    if all_articles == "":
        raise ValueError("No articles found.")
    
    return all_articles


#extract entity names from article
def extract_article_names(article: spacy.tokens.doc.Doc, sim_threshold = 0.8): 
    # labels of entities that we dont want as nodes
    unwanted_label = ["DATE", "TIME", "ORDINAL", "CARDINAL"]
    #in the end, nodes will contain all relevant entities and their labels
    nodes = {}
    #the similarity dict has an entry for each entity in the network. These entities are dictionaries themselves, which include the similarity between the entity and all other entities.
    # Example:
    # similarities = {"Donald Trump": {"Barack Obama": 0.7, "Angela Merkel": 0.5}, 
    #                 "Barack Obama": {"Donald Trump": 0.7, "Angela Merkel": 0.6}, 
    #                 "Angela Merkel": {"Donald Trump": 0.5, "Barack Obama": 0.6}}
    similarities = {}

    for ent in article.ents:
        #remove all 's at the end of entity labels 
        base_form_ent = re.sub(r"'s$", '', ent.text)
        if (base_form_ent not in nodes): 
            nodes[base_form_ent] = {"label": ent.label_}
        if (ent.label_ not in unwanted_label) and (ent.has_vector):
            similarities[base_form_ent] = {}
            for other_ent in article.ents:
                base_form_other_ent = re.sub(r"'s$", '', other_ent.text)
                # add base_form_ent and base_form_other_ent to similarities
                if (other_ent.label_ not in unwanted_label) and (other_ent.has_vector) and (base_form_ent != base_form_other_ent):
                    if base_form_ent in similarities:
                        similarities[base_form_ent][base_form_other_ent] = ent.similarity(other_ent)
                    else:
                        similarities[base_form_ent] = {base_form_other_ent : ent.similarity(other_ent)}
    return reduce_nodes(similarities, nodes, sim_threshold)


# remove nodes which are very similar to another node which also contains their label  
def reduce_nodes(similarities: dict, nodes: dict, sim_threshold = 0.8):

    extracted_entities = []
    for entity in similarities:
        #append entity name to list if the name is not part of the name of another entity and they have a high similarity
        append = True
        for other_entity in similarities[entity]:
            if entity in other_entity:
                try:
                    if similarities[entity][other_entity] > sim_threshold:
                        append = False
                        break 
                except KeyError:
                    raise KeyError("There doesn't seem to be a similarity score for {} and {}.".format(entity, other_entity))                  
        for other_entity in extracted_entities:
            if entity.lower() == other_entity.lower():
                append = False
                break
        if append == True:
            extracted_entities.append(entity)
    #only keep those nodes which are in extracted_entities        
    nodes = {key: nodes[key] for key in extracted_entities}
    
    return extracted_entities, similarities, nodes


#delete all entries in similarity dict where at least one of the entities is not in extracted_entities
def clean_similarities(similarities: dict, extracted_entities: list):

    try:
        similarities =  { key: similarities[key] for key in extracted_entities }
    except KeyError:
        raise KeyError("Some key doesn't seem to be in similarities, but appears in extracted_entities!")
    for entity in similarities:
        deletelist = []
        for other_entity in similarities[entity]: 
            if other_entity not in extracted_entities:
                deletelist.append(other_entity)
        for unwanted_key in deletelist: 
            del similarities[entity][unwanted_key]
    #divide the word similarity by three for the similarity measure
    for ent in similarities:
        for other_ent in similarities[ent]:
            similarities[ent][other_ent] /= 3 
    
    return similarities


#add nodes to the network, including their labels and plot their sizes accordingly
def add_nodes(network: nx.Graph(), nodes: dict, location_size, company_size, entity_size):
    
    for entity in nodes:
        try:
            if nodes[entity]["label"] == "GPE" or nodes[entity]["label"] == "LOC":
                network.add_node(entity, size = location_size)

            elif nodes[entity]["label"] == "ORG":
                network.add_node(entity, size = company_size)
            else:
                network.add_node(entity, size = entity_size)
        except TypeError:
            raise TypeError("The node {} doesn't seem to have a label!".format(entity))
    
    return network


#add an edge between all nodes above a certain similarity threshold to the network
def add_edges_with_high_sim(network: nx.Graph(), extracted_entities: list, similarities: dict, high_sim_value : float):
    #edges which don't appear in the network already. Those edges are colored red.
    new_edges = []
    #already_added will contain all edges which have been added to the network before.
    #those edges are going to be colored in black
    already_added = []
    
    for entity in extracted_entities:
        for other_entity in extracted_entities:
            if entity != other_entity:
                try:
                    entity_sim = similarities[entity][other_entity]
                except KeyError:
                    raise KeyError("There doesn't seem to be a similarity score for {} and {}.".format(entity, other_entity))
                except TypeError:
                    raise TypeError("The similarity dict has an unexpected format.")
                else:
                    if entity_sim >= high_sim_value:
                        if (entity, other_entity) in network.edges:
                            already_added.append((entity, other_entity))
                        else:
                            new_edges.append((entity, other_entity))

    network.add_edges_from(new_edges, color = "red", title = "word similarity")
    network.add_edges_from(already_added, color = "black", title = "word similarity")
    
    return network


#returns a list of all overlapping entries in two lists (in this case: lists of entity_ids)
def find_overlapping_ids(entity_ids: list, other_entity_ids: list):
    overlap = list(set(entity_ids) & set(other_entity_ids))
    return overlap


#determines a list of babelnet_ids for an entity
def find_babelnet_ids(entity: str, api: py_babelnet.calls.BabelnetAPI):
    bn_ids = []
    
    try:
        entity_ids = api.get_synset_ids(lemma = entity, searchLang = "EN", targetLang = "EN")
    except ValueError :
        raise ValueError("Not enough BabelCoins to determine entity IDs.")
    except socket.error:
        print("Connection Error - please check your internet connection and try again.")
        raise
    
    if entity_ids:
        #message occurs when running out of babel coins or giving an invalid api-key
        if "message" not in entity_ids:
            for entity_id in entity_ids:
                bn_ids.append(entity_id["id"])
        else:
            raise ValueError(entity_ids["message"] + " Please consider checking your .env as well.")
    return bn_ids


#finds all ID belonging to the extracted entities in the BabelNet Network and saves them in bn_ids
#additionally extracts all edges belonging to those IDs and saves them in bn_info
def get_babelnet_info(extracted_entities: list, api: py_babelnet.calls.BabelnetAPI, bn_info: dict, bn_ids: dict):
    #find all IDS for the entity name in the babelnet network
    for entity in extracted_entities:
        if entity not in bn_ids:
            bn_ids[entity] = find_babelnet_ids(entity, api)

        # find all outgoing edges from the entities in the babelnet network
        if (entity not in bn_info) and (entity in bn_ids):
            entity_ids = bn_ids[entity]
            entity_edges = pd.DataFrame(columns = ["target", "name"])
            
            for entity_id in entity_ids:
                try:
                    edges = api.get_outgoing_edges(id = entity_id)
                    edges_df = pd.DataFrame(edges)
                except ValueError:
                    raise ValueError("Not enough BabelCoins to determine outgoing edges.")
                except socket.error:
                    print("Connection Error - please check your internet connection and try again.")
                    raise
                if not edges_df.empty:
                    edges_df["name"] = (pd.DataFrame(edges_df["pointer"].tolist()))["name"]
                    edges_df = edges_df.drop(columns = ["weight", "language", "pointer", "normalizedWeight"])
                    #edges_df now only contains the columns "target" and "name", where "name" is the edge-label 
                    entity_edges = pd.concat([entity_edges, edges_df])
            #bn_info is now a dictionary of dataframes containing all edges found in the babelnet network
            bn_info[entity] = entity_edges  
    
    return bn_info, bn_ids


#finds all edges between two of our entities from the BN-Network 
def find_babelnet_edges(bn_ids: dict, bn_info: dict):
    edges = pd.DataFrame(columns = ["from", "to", "label"]) 
    for entity in bn_ids:
        for entity_id in bn_ids[entity]:
            for other_entity in bn_info:
                if entity != other_entity:
                    other_entity_edges = bn_info[other_entity]
                    #add an edge if any id of entity is a target in the edge-dataframe of the other entity 
                    try:                   
                        if other_entity_edges["target"].str.contains(entity_id).any():
                            incident_edges = other_entity_edges.loc[(other_entity_edges["target"] == entity_id)]
                            incident_edges = incident_edges.drop_duplicates()
                            incident_edges = incident_edges.reset_index()
                            #if there is any edge with a different label than "Semantically related form", remove all edges with the label "Semantically related form"
                            if not incident_edges["name"].str.contains("Semantically related form").all():
                                incident_edges = incident_edges[incident_edges["name"] != "Semantically related form"]
                            outg_edges = pd.DataFrame(columns = ("from", "to", "label"))
                            outg_edges["from"] = [other_entity]*len(incident_edges)
                            outg_edges["to"] = [entity]*len(incident_edges)
                            outg_edges["label"] = incident_edges["name"]
                            edges = pd.concat([edges, outg_edges], ignore_index = True)
                    except TypeError:
                        raise TypeError("bn_info seems to be in a wrong format. It should be a dict of key - dataframe pairs.")
                    except KeyError:
                        raise KeyError("The dataframes in bn_info need to have columns called 'name' and 'target' ")

    edges = edges.drop_duplicates()
    edges = edges.reset_index()
    return edges


#add the babelnet edges to the network and update similarity measure
def add_babelnet_edges(network: nx.Graph(), edges: pd.DataFrame, similarity_measure: dict, unlabelled_edges: bool):
    #use sim_update to update the similarity_measure: add 1/3 if an edge in the babel-net network exists or if an edge with a label exists (if unlabelled_edges = false)
    new_edges = []
    sim_update = {}
    if not unlabelled_edges:
        edges = edges[edges["label"] != "Semantically related form"]
    for index, row in edges.iterrows():
        add = True
        try:
            if ([row["from"], row["to"]]) not in network.edges:
                color = "red"
                new_edges.append((row["from"], row["to"]))
                new_edges.append((row["to"], row["from"]))
            else:
                if((row["from"], row["to"]) not in new_edges):
                    color = "black"
                else:
                    add = False
        except KeyError:
            raise KeyError("The edges need to have columns called 'from' and 'to'.")
        #only add new edges
        if add == True:
            try:
                network.add_edge(row["from"], row["to"], title = row["label"], color = color)
            except KeyError:
                raise KeyError("The edge-dataframe needs to have columns called 'from', 'to' and 'label'.")
        
        if row["from"] in sim_update:
            if row["to"] not in sim_update[row["from"]]:
                sim_update[row["from"]].append(row["to"])
        else:
            sim_update[row["from"]] = [row["to"]]
        
        if row["to"] in sim_update:
            if row["from"] not in sim_update[row["to"]]:
                sim_update[row["to"]].append(row["from"])
        else:
            sim_update[row["to"]] = [row["from"]]

    # update similarity measure
    for entity in sim_update:
        if entity not in similarity_measure:
            similarity_measure[entity] = {}
        for other_entity in sim_update[entity]:
            if other_entity not in similarity_measure[entity]:
                similarity_measure[entity][other_entity] = 1/3
            else:
                similarity_measure[entity][other_entity] += 1/3        
    
    return network, similarity_measure


#creates a list of sentences from a text
def split_sentences(text: str):
    # split sentences and questions
    text = re.split('[.?!]', text)
    clean_sent = []
    for sent in text:
        clean_sent.append(sent)
    return clean_sent


#extract entities from the wikipedia-summary of the article-entities and calculate their similarity
def calculate_wikipedia_similarity(nlp: spacy.language, extracted_entities: list, num_sentences = 3):
    
    unwanted_label = ["DATE", "ORDINAL", "CARDINAL", "PERCENT"]
    #wiki_ents is a dictionary where the keys are the article entities and the values are lists of all entities which appear in the first (num_sentences) of their wikipedia-summary
    wiki_ents = {}
    #wiki_sentences is a dictionary where the keys are the article entities and the values are lists of the split sentences of their wikipedia-summaries.
    wiki_sentences = {}
    #sim_wiki is a dictionary with entity keys and dictionaries as values. These dictionaries contain the wikipedia-similarity between the pairs of nodes.
    sim_wiki = {}
    for entity in extracted_entities:
        wiki_ents[entity] = []
        wiki_sentences[entity] = []
        try:
            descr = wikipedia.summary(entity, sentences = num_sentences)
        except (wikipedia.exceptions.DisambiguationError, wikipedia.exceptions.PageError):
            try:
                descr = wikipedia.summary(wikipedia.suggest(entity), sentences = num_sentences)
                print("Wikipedia error for entity ", entity, " use ", wikipedia.suggest(entity), " instead")
            except (wikipedia.exceptions.DisambiguationError, wikipedia.exceptions.PageError, TypeError, ValueError):
                print("Wikipedia disambiguation error occured for {}, skipping this entity".format(entity))
                continue
            except socket.error:
                print("Connection Error - please check your internet connection and try again.")
                raise
        except socket.error:
            print("Connection Error - please check your internet connection and try again.")
            raise

        split_descr = split_sentences(descr)
        split_descr = split_descr[:num_sentences]
        wiki_sentences[entity] = split_descr
        
        #calculate similarity for overlapping entities and store the occuring entities in wiki_ents
        sentence_number = 1
        for sentence in split_descr:
            entity_descr = nlp(sentence)
            for ent in entity_descr.ents:
                if (ent.label_ not in unwanted_label) and (ent.text != entity):
                    wiki_ents[entity].append(ent.text)
                    if entity not in sim_wiki:
                        sim_wiki[entity] = {ent.text: 1/sentence_number}
                    else:
                        if ent not in sim_wiki[entity]:
                            sim_wiki[entity][ent.text] = 1/sentence_number
            sentence_number = sentence_number + 1
        for item in wiki_ents:
            wiki_ents[item] = list(dict.fromkeys(wiki_ents[item]))
    
    return wiki_ents, sim_wiki

#add the similarity score of two entities to the similarity measure dict
def add_similarity(entity: str, other_entity: str, sim_score: float, similarity_measure:dict):

    if entity in similarity_measure:
        try:
            if other_entity in similarity_measure[entity]:
                similarity_measure[entity][other_entity] += sim_score
            else:
                similarity_measure[entity][other_entity] = sim_score
        except TypeError:
            print("The similarity dict has an unexpected format.")
            raise TypeError() 
    else:
        similarity_measure[entity] = {other_entity : sim_score}
    return similarity_measure


#add the direct edges from wikipedia to the network and update the similarity measure
def find_direct_wikipedia_edges(network: nx.Graph(), wiki_ents: dict, extracted_entities: list, similarity_measure: dict, sim_wiki: dict, api: py_babelnet.calls.BabelnetAPI):
    new_edges = []
    already_added = []
    #dict where each lists of ids are assigned to entities
    wiki_ids = {}
    for entity in wiki_ents:
        for summ_entity in wiki_ents[entity]:
            if summ_entity != entity:
                if summ_entity in extracted_entities:
                    if ([entity, summ_entity]) not in network.edges:
                        new_edges.append((entity, summ_entity))
                    else:
                        already_added.append((entity, summ_entity))
                    # update the similarity measure with the max. wikipedia-similarity of the pair of entities
                    try:
                        if summ_entity not in sim_wiki:
                            sim_score = sim_wiki[entity][summ_entity]/3
                        elif entity not in sim_wiki[summ_entity]:
                            sim_score = sim_wiki[entity][summ_entity]/3
                        else:
                            sim_score = max(sim_wiki[entity][summ_entity], sim_wiki[summ_entity][entity])/3
                    except TypeError:
                        print("The similarity dict has an unexpected format.")
                        raise TypeError() 

                    similarity_measure = add_similarity(entity, summ_entity, sim_score, similarity_measure) 
                    similarity_measure = add_similarity(summ_entity, entity, sim_score, similarity_measure)

                #create wiki_ids: babelnet IDs of the wikipedia entities of the article of summ_entity
                #stop if there are not enough babel coins to find the wikipedia-ids -> this will be excepted in main(), where we will then continue without the wikipedia-edges
                elif summ_entity not in wiki_ids:
                    try:
                        wiki_ids[summ_entity] = find_babelnet_ids(summ_entity, api)
                    except ValueError:
                        print("Not enough BabelCoins - Continuing without the Wikipedia edges.")
                        raise ValueError()
                    except socket.error:
                        print("Connection Error - please check your internet connection and try again.")
                        raise
       
    network.add_edges_from(new_edges, color = "red", title = "wikipedia direct")
    network.add_edges_from(already_added, color = "black", title = "wikipedia direct")
    return wiki_ids, similarity_measure, network


#compare the BN-Ids of the wikipedia entities to those of the BN-Entities and substitute those with overlapping IDs.
def remove_duplicate_wikipedia_ents(bn_ids: dict, wiki_ents: dict, wiki_ids: dict):
    #bn_id_list includes all bnIds
    bn_id_list = []
    # wikiIds_retrieved: dict. This has article entities as keys and lists of ids as values. 
    # Those ID-lists are obtained by comparing the BN_IDs of wikipedia-entities to the article BN-IDs. If there is an overlap, both entities are considered equal.
    # We can then substitute the wikipedia-entities with overlapping IDs for the article entities
    wiki_ids_retrieved = {}
    # wiki_entities_retrieved has wikipedia-entities as keys and lists of the article-entities with overlapping IDs as values.
    # we can later use this to connect entities to the retrieved article-entities instead of the wikipedia-entities
    wiki_entities_retrieved = {}
    for entity in bn_ids:
        bn_id_list.extend(bn_ids[entity])
    for entity in wiki_ents:
        for summ_ent in wiki_ents[entity]:
            added = False
            if summ_ent in wiki_ids:
                #check if the wikipedia entity has overlapping IDs with one of the BN-Entities
                if any(summ_ent_id in bn_id_list for summ_ent_id in wiki_ids[summ_ent]):
                    for label in bn_ids:
                        if label != entity:
                            #if this is the overlapping entity
                            if any(summ_ent_id in bn_ids[label] for summ_ent_id in wiki_ids[summ_ent]):
                                added = True
                                if label not in wiki_ids_retrieved:
                                    wiki_ids_retrieved[label] = set(wiki_ids[summ_ent])
                                else:                          
                                    wiki_ids_retrieved[label].update(set(bn_ids[label]))
                                if entity not in wiki_entities_retrieved:
                                    wiki_entities_retrieved[entity] = {label}
                                else:
                                    wiki_entities_retrieved[entity].add(label)
                if not added:
                    if entity in wiki_entities_retrieved:
                        wiki_entities_retrieved[entity].add(summ_ent)
                    else:
                        wiki_entities_retrieved[entity] = {summ_ent}
                    if entity in wiki_ids_retrieved:
                        wiki_ids_retrieved[entity].update(set(wiki_ids[summ_ent]))
                    else:
                        wiki_ids_retrieved[entity] = set(wiki_ids[summ_ent])
                        
    return wiki_ids_retrieved, wiki_entities_retrieved


#add the indirect wikipedia edges to the network
def find_indirect_wikipedia_edges(network: nx.Graph(), wiki_ents_retrieved: dict, wikiIds_retrieved: dict):
    new_edges = []
    already_added = []
    
    for entity in wiki_ents_retrieved:
        for other_entity in wiki_ents_retrieved:
            if entity != other_entity:
                for summ_entity in wiki_ents_retrieved[entity]:
                    for summ_other_entity in wiki_ents_retrieved[other_entity]:
                        #if both entities have an overlapping wikipedia-summary-entity, add an indirect edge to the network
                        if summ_entity == summ_other_entity and summ_entity != entity and summ_entity != other_entity:
                            if (entity, summ_entity) in network.edges:
                                already_added.append((entity, summ_entity))
                            else:
                                new_edges.append((entity, summ_entity))
                            if (other_entity, summ_entity) in network.edges:
                                already_added.append((other_entity, summ_entity))
                            else:
                                new_edges.append((other_entity, summ_entity))
                        else:
                            #if the wikipedia-summary entities have overlapping BabelNet IDs, treat them as equal and add an indirect edge to the network
                            if (summ_entity in wikiIds_retrieved) and (summ_other_entity in wikiIds_retrieved):
                                if any(summ_ids in wikiIds_retrieved[summ_other_entity] for summ_ids in wikiIds_retrieved[summ_entity]) and summ_entity != entity and summ_entity != other_entity:
                                    if (entity, summ_entity) in network.edges:
                                        already_added.append((entity, summ_entity))
                                    else:
                                        new_edges.append((entity, summ_entity))
                                    if (other_entity, summ_entity) in network.edges:
                                        already_added.append((other_entity, summ_entity))
                                    else:
                                        new_edges.append((other_entity, summ_entity))
                                        
    network.add_edges_from(new_edges, color = "red", title = "wikipedia indirect")
    network.add_edges_from(already_added, color = "black", title = "wikipedia indirect")
    return network


#reset the colors of the edges of the network
def reset_edgecolors(network: nx.Graph()):
    for edge in network.edges:
        network.edges[edge].pop("color", None)
    return network


#compute the wikipedia networks or skip them if not enough babelcoins
def compute_wikipedia_networks(network: nx.Graph(), bn_ids: dict, extracted_entities: list, similarity_measure: dict, api: py_babelnet.calls.BabelnetAPI, nlp: spacy.language, wikipedia_sentences: int):
    wiki_ents, sim_wiki = calculate_wikipedia_similarity(nlp, extracted_entities, wikipedia_sentences)
    pyvis_network = Network()
    try:
        wiki_ids, similaritiy_measure, network = find_direct_wikipedia_edges(network, wiki_ents, extracted_entities, similarity_measure, sim_wiki, api)
    except ValueError:
        raise ValueError
    else:
        pyvis_network.from_nx(network)
        pyvis_network.save_graph("wikipedia_network_direct_{}.html".format(time.strftime("%Y%m%d-%H%M%S")))
        network = reset_edgecolors(network)
        wikiIds_retrieved, wiki_entities_retrieved = remove_duplicate_wikipedia_ents(bn_ids, wiki_ents, wiki_ids)
        network = find_indirect_wikipedia_edges(network, wiki_entities_retrieved, wikiIds_retrieved)
        pyvis_network = Network()
        pyvis_network.from_nx(network)
        pyvis_network.save_graph("wikipedia_network_indirect_{}.html".format(time.strftime("%Y%m%d-%H%M%S")))
        network = reset_edgecolors(network)
        return network


#add the similarity of two nodes as a value to the edges between two nodes
def add_similarity_to_edges(network: nx.Graph(), similarity_measure: dict): 
    for edge in network.edges:
        network.edges[edge]["weight"] = 0
    for entity in similarity_measure:
        for other_entity in similarity_measure[entity]:
            if ([entity, other_entity] in network.edges):
                network.edges[entity, other_entity]["weight"] = similarity_measure[entity][other_entity]
                if network.edges[entity, other_entity]["weight"] > 0:
                    weight_label = ", weight: {:.2f}".format(similarity_measure[entity][other_entity])
                    if "title" in network.edges[entity, other_entity] and not pd.isna(network.edges[entity, other_entity]["title"]):
                        if weight_label not in network.edges[entity, other_entity]["title"]:
                            network.edges[entity, other_entity]["title"] += weight_label
                    else:
                        network.edges[entity, other_entity]["title"] = weight_label
    return network


#filter network for edges with a similarity larger than a given sim_threshold
def filter_similarity(network: nx.Graph(), sim_threshold = 0.8):
    sim_network = nx.Graph([(u, v, properties) for u, v, properties in network.edges(data = True) if "weight" in properties and properties["weight"] > sim_threshold])
    return sim_network


#filter network for nodes with certain labels
def filter_node_labels(network: nx.Graph(), labels: list):
    filtered_network = nx.Graph([(u, v, properties) for u, v, properties in network.edges(data = True) if any(label in u.lower() or label in v.lower() for label in labels)])
    return filtered_network


#filter network for edges with certain labels
def filter_edge_labels(network: nx.Graph(), labels: list):
    filtered_network = nx.Graph([(u, v, properties) for u, v, properties in network.edges(data = True) if "title" in properties and not pd.isna(properties["title"]) and any(label in properties['title'].lower() for label in labels)])
    return filtered_network


#interactively filters the network
def filtering_process(network: nx.Graph()):
    exit = False

    while exit == False:
        filter_type = input("Do you wish to filter your network? \n [n] - filter nodes with a certain label \n [e] - filter edges with a certain label \n [s] - filter edges over a certain similarity threshold \n [other] - quit \n").lower()
        #differentiate between the different filter types
        if filter_type == "n":
            node_labels = [item for item in input("Please enter the node labels for which you want to filter the network: \n").split()]
            filtered_node_network = filter_node_labels(network, node_labels)
            #dont save the network if it is empty
            if len(filtered_node_network.nodes) == 0:
                print("No nodes found for labels {}.\n".format(node_labels))
            else:
                pyvis_network = Network()
                pyvis_network.from_nx(filtered_node_network)
                pyvis_network.save_graph("filtered_node_nw_{}.html".format(time.strftime("%Y%m%d-%H%M%S")))

        elif filter_type == "e":
            edge_labels = [item for item in input("Please enter the edge labels for which you want to filter the network: \n").split()]
            filtered_edge_network = filter_edge_labels(network, edge_labels)
            if (len(filtered_edge_network.edges) == 0):
                print("No edges found for labels {}!\n".format(edge_labels))
            else:
                pyvis_network = Network()
                pyvis_network.from_nx(filtered_edge_network)
                pyvis_network.save_graph("filtered_edge_nw_{}.html".format(time.strftime("%Y%m%d-%H%M%S")))

        elif filter_type == "s":
            try:
                similarity_score = [float(item) for item in (input("Please enter the similarity thresholds for which you want to filter the network: \n").split())]
            except ValueError:
                print("Please enter a valid similarity threshold (for example 0.6)")
            else:
                for sim in similarity_score:
                    sim_network = filter_similarity(network, sim)
                    if len(sim_network.edges) == 0:
                        print("No edges found for similarity score {}.\n".format(sim))
                    else:
                        pyvis_network = Network()
                        pyvis_network.from_nx(sim_network)
                        pyvis_network.save_graph("filtered_similarity_{}_nw_{}.html".format(sim, time.strftime("%Y%m%d-%H%M%S")))
        else:
            exit = True


def __main__():

    nlp = spacy.load('en_core_web_lg')
    
    #initialize babel net api 
    api = init_bn_api()
    bn_info = {}
    bn_ids = {}
    similarity_measure = {}
    network = nx.Graph()
    pyvis_network = Network()
    sim_threshold, wikipedia_sentences, location_size, org_size, entity_size, unlabelled_edges = init_values()
    articles = ""
    
    if len(sys.argv) > 1:
        paths = sys.argv[1:]

    else: 
        paths = [item for item in input("Please enter a path to an article: \n").split()]
    
    while articles == "":
        try:
            articles = read_articles(paths)
        except ValueError:
            paths = [item for item in input("No articles found. Please enter a valid path to an article. Enter 'quit' if you wish to stop.\n").split()]
        if "quit" in paths:
            return   
    nlp_article = nlp(articles)

    #create nodes
    extracted_entities, similarities, nodes = extract_article_names(nlp_article)
    similarity_measure = clean_similarities(similarities, extracted_entities)
    network = add_nodes(network, nodes, location_size, org_size, entity_size)
    #add edges between nodes with similarity larger than sim_threshold
    network = add_edges_with_high_sim(network, extracted_entities, similarity_measure, sim_threshold)
    pyvis_network.from_nx(network)
    pyvis_network.save_graph("wordsimilarity_network_{}.html".format(time.strftime("%Y%m%d-%H%M%S")))
    network = reset_edgecolors(network)

    #collect info from babelnet and find edges in the network
    bn_info, bn_ids = get_babelnet_info(extracted_entities, api, bn_info, bn_ids)
    edges = find_babelnet_edges(bn_ids, bn_info)
    #Create the babelnet network
    network, similarity_measure = add_babelnet_edges(network, edges, similarity_measure, unlabelled_edges)
    pyvis_network = Network()
    pyvis_network.from_nx(network)
    pyvis_network.save_graph("babelnet_network_{}.html".format(time.strftime("%Y%m%d-%H%M%S")))
    network = reset_edgecolors(network)
    #If we dont run out of babel-coins, find the wikipedia-edges. Else, skip this part.
    try:
        network = compute_wikipedia_networks(network, bn_ids, extracted_entities, similarity_measure, api, nlp, wikipedia_sentences)
    except ValueError :
        pass

    #add the similarity score to the edge-labels
    network = add_similarity_to_edges(network, similarity_measure)
    pyvis_network = Network()
    pyvis_network.from_nx(network)
    pyvis_network.save_graph("weighted_network_{}.html".format(time.strftime("%Y%m%d-%H%M%S")))
    network = reset_edgecolors(network)

    filtering_process(network)


if __name__ == "__main__":
    __main__()