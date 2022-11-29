from omegaconf import OmegaConf

config = OmegaConf.load('configs/Query/Query_entity_marker_sentence4_reversed.yaml')

query = config.data.query

sentence = '강낭콩팥쥐'

print(eval(query))
