def parse_triples(relation_set):
    subjects, predicates, objects = set(), set(), set()
    for o, p, s in relation_set:
        subjects.add(s)
        predicates.add(p)
        objects.add(o)
    return objects, predicates, subjects


def parse_types_list(relation_set):
    subjects, types = list(), list()
    for s, t in relation_set:
        subjects.append(s)
        types.append(t)
    return subjects, types


def parse_types(relation_set):
    subjects, types = set(), set()
    for s, t in relation_set:
        subjects.add(s)
        types.add(t)
    return subjects, types

# 将一个三元组转为id，存储一个kg所需的各种数据
class KG:
    def __init__(self, relation_triples, attribute_triples):
        self.type_dict = None
        self.r_dict = None
        self.h_dict = None
        self.t_dict = None
        self.ht_dict = None
        self.test_relation_triples_list = None
        self.test_relation_triples_set = None
        self.valid_relation_triples_list = None
        self.valid_relation_triples_set = None
        self.entities_set, self.entities_list = None, None
        self.relations_set, self.relations_list = None, None
        self.attributes_set, self.attributes_list = None, None
        self.type_set, self.type_list = None, None
        self.entities_num, self.relations_num, self.attributes_num, self.type_num = None, None, None, None
        self.relation_triples_num, self.attribute_triples_num = None, None
        self.local_relation_triples_num, self.local_attribute_triples_num = None, None
        self.valid_et_ent, self.valid_et_type, self.test_et_ent, self.test_et_type = None, None, None, None
        self.entities_id_dict = None
        self.relations_id_dict = None
        self.attributes_id_dict = None
        self.train_et_list, self.valid_et_list, self.test_et_list = None, None, None
        self.rt_dict, self.hr_dict = None, None
        self.entity_relations_dict = None
        self.entity_attributes_dict = None
        self.av_dict = None

        self.sup_relation_triples_set, self.sup_relation_triples_list = None, None
        self.sup_attribute_triples_set, self.sup_attribute_triples_list = None, None

        self.relation_triples_set = None
        self.attribute_triples_set = None
        self.relation_triples_list = None
        self.attribute_triples_list = None

        self.local_relation_triples_set = None
        self.local_relation_triples_list = None
        self.local_attribute_triples_set = None
        self.local_attribute_triples_list = None

        self.set_relations(relation_triples)
        self.set_attributes(attribute_triples)

        print()
        print("KG statistics:")
        print("Number of entities:", self.entities_num)
        print("Number of relations:", self.relations_num)
        print("Number of attributes:", self.attributes_num)
        print("Number of relation triples:", self.relation_triples_num)
        print("Number of attribute triples:", self.attribute_triples_num)
        print("Number of local relation triples:", self.local_relation_triples_num)
        print("Number of local attribute triples:", self.local_attribute_triples_num)
        print()

    def set_relations(self, relation_triples):
        self.relation_triples_set = set(relation_triples)
        self.relation_triples_list = list(self.relation_triples_set)
        self.local_relation_triples_set = self.relation_triples_set
        self.local_relation_triples_list = self.relation_triples_list

        heads, relations, tails = parse_triples(self.relation_triples_set)
        self.entities_set = heads | tails
        self.relations_set = relations
        self.entities_list = list(self.entities_set)
        self.relations_list = list(self.relations_set)
        self.entities_num = len(self.entities_set)
        self.relations_num = len(self.relations_set)
        self.relation_triples_num = len(self.relation_triples_set)
        self.local_relation_triples_num = len(self.local_relation_triples_set)
        self.generate_relation_triple_dict()
        self.parse_relations()
        
    def set_valid_relations(self, valid_relations):
        self.valid_relation_triples_set = set(valid_relations)
        self.valid_relation_triples_list = list(self.valid_relation_triples_set)
        heads, relations, tails = parse_triples(self.valid_relation_triples_set)
        enset = heads | tails
        self.entities_set |= enset
        self.relations_set |= relations
        self.entities_list = list(self.entities_set)
        self.relations_list = list(self.relations_set)
        self.entities_num = len(self.entities_set)
        self.relations_num = len(self.relations_set)

    def set_test_relations(self, test_relations):
        self.test_relation_triples_set = set(test_relations)
        self.test_relation_triples_list = list(self.test_relation_triples_set)
        heads, relations, tails = parse_triples(self.test_relation_triples_set)
        enset = heads | tails
        self.entities_set |= enset
        self.relations_set |= relations
        self.entities_list = list(self.entities_set)
        self.relations_list = list(self.relations_set)
        self.entities_num = len(self.entities_set)
        self.relations_num = len(self.relations_set)
        
    def set_attributes(self, attribute_triples):
        self.attribute_triples_set = set(attribute_triples)
        self.attribute_triples_list = list(self.attribute_triples_set)
        self.local_attribute_triples_set = self.attribute_triples_set
        self.local_attribute_triples_list = self.attribute_triples_list

        entities, attributes, values = parse_triples(self.attribute_triples_set)
        self.attributes_set = attributes
        self.attributes_list = list(self.attributes_set)
        self.attributes_num = len(self.attributes_set)

        # add the new entities from attribute triples
        self.entities_set |= entities
        self.entities_list = list(self.entities_set)
        self.entities_num = len(self.entities_set)

        self.attribute_triples_num = len(self.attribute_triples_set)
        self.local_attribute_triples_num = len(self.local_attribute_triples_set)
        self.generate_attribute_triple_dict()
        self.parse_attributes()

    def generate_attribute_triple_dict(self):
        self.av_dict = dict()
        for h, a, v in self.local_attribute_triples_list:
            av_set = self.av_dict.get(h, set())
            av_set.add((a, v))
            self.av_dict[h] = av_set
        print("Number of av_dict:", len(self.av_dict))

    def generate_relation_triple_dict(self):
        self.rt_dict, self.hr_dict = dict(), dict()
        self.r_dict, self.h_dict, self.t_dict = dict(), dict(), dict()
        for h, r, t in self.local_relation_triples_list:
            rt_set = self.rt_dict.get(h, set())
            rt_set.add((r, t))
            self.rt_dict[h] = rt_set
            hr_set = self.hr_dict.get(t, set())
            hr_set.add((h, r))
            self.hr_dict[t] = hr_set

            r_set = self.r_dict.get((h, t), set())
            r_set.add(r)
            self.r_dict[(h, t)] = r_set

            h_set = self.h_dict.get((r, t), set())
            h_set.add(h)
            self.h_dict[(r, t)] = h_set

            t_set = self.t_dict.get((h, r), set())
            t_set.add(t)
            self.t_dict[(h, r)] = t_set
        print("Number of rt_dict:", len(self.rt_dict))
        print("Number of hr_dict:", len(self.hr_dict))
        print("Number of h_dict:", len(self.h_dict))
        print("Number of r_dict:", len(self.r_dict))
        print("Number of t_dict:", len(self.t_dict))

    def parse_relations(self):
        self.entity_relations_dict = dict()
        for ent, attr, _ in self.local_relation_triples_set:
            attrs = self.entity_relations_dict.get(ent, set())
            attrs.add(attr)
            self.entity_relations_dict[ent] = attrs
        print("entity relations dict:", len(self.entity_relations_dict))

    def parse_attributes(self):
        self.entity_attributes_dict = dict()
        for ent, attr, _ in self.local_attribute_triples_set:
            attrs = self.entity_attributes_dict.get(ent, set())
            attrs.add(attr)
            self.entity_attributes_dict[ent] = attrs
        print("entity attributes dict:", len(self.entity_attributes_dict))

    def set_type_list(self, train_type_id_dict, valid_type_id_dict, test_type_id_dict):
        self.train_et_list = train_type_id_dict
        self.valid_et_list = valid_type_id_dict
        self.test_et_list = test_type_id_dict
        print("entity type dict:", len(self.train_et_list))
        _, self.type_set = parse_types(self.train_et_list)
        self.type_list = list(self.type_set)
        self.type_num = len(self.type_list)
        print("entity type num:", self.type_num)
        self.valid_et_ent, self.valid_et_type = parse_types_list(self.valid_et_list)
        self.test_et_ent, self.test_et_type = parse_types_list(self.test_et_list)
        self.type_dict = dict()
        for h, t in self.train_et_list:
            type_set = self.type_dict.get((h, 0), set())
            type_set.add(t)
            self.type_dict[(h, 0)] = type_set

        print("Number of type_dict:", len(self.type_dict))

    def set_id_dict(self, entities_id_dict, relations_id_dict, attributes_id_dict):
        self.entities_id_dict = entities_id_dict
        self.relations_id_dict = relations_id_dict
        self.attributes_id_dict = attributes_id_dict

    def add_sup_relation_triples(self, sup_triples):
        self.sup_relation_triples_set = set(sup_triples)
        self.sup_relation_triples_list = list(self.sup_relation_triples_set)
        self.relation_triples_set |= sup_triples
        self.relation_triples_list = list(self.relation_triples_set)
        self.relation_triples_num = len(self.relation_triples_list)

    def add_sup_attribute_triples(self, sup_triples):
        self.sup_attribute_triples_set = set(sup_triples)
        self.sup_attribute_triples_list = list(self.sup_attribute_triples_set)
        self.attribute_triples_set |= sup_triples
        self.attribute_triples_list = list(self.attribute_triples_set)
        self.attribute_triples_num = len(self.attribute_triples_list)
