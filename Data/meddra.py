import os
from collections import ChainMap
from anytree import Node, RenderTree, findall_by_attr, Walker
from tqdm import tqdm
import pandas as pd
#Класс для парсиннга medDRA и UMLS, обеспечивает линкинг кодов друг к другу
#Структура medDRA по вложенности llt->pt->hlt->hlgt->soc
class MeddraParser():
    def __init__(self, meddra_path, umls_path=None, lang='ENG'):
        assert lang in ['ENG', 'RUS'], "Choose 'RUS' or 'ENG'"
        self.meddra_path_pt = os.path.join(meddra_path, 'ascii-241', 'pt.asc')
        self.meddra_path_llt = os.path.join(meddra_path, 'ascii-241', 'llt.asc')
        self.meddra_path_hlt = os.path.join(meddra_path, 'ascii-241', 'hlt.asc')
        self.meddra_path_hlt_to_pt = os.path.join(meddra_path, 'ascii-241', 'hlt_pt.asc')
        self.meddra_path_hlgt = os.path.join(meddra_path, 'ascii-241', 'hlgt.asc')
        self.meddra_path_hlgt_to_hlt = os.path.join(meddra_path, 'ascii-241', 'hlgt_hlt.asc')
        self.meddra_path_soc = os.path.join(meddra_path, 'ascii-241', 'soc.asc')
        self.meddra_path_soc_to_hlgt = os.path.join(meddra_path, 'ascii-241', 'soc_hlgt.asc')
        self.language = lang
        
        self.ptcodes = []
        self.lltcodes = []
        self.hltcodes = []
        self.hlgtcodes = []
        self.soccodes = []
        
        self.lltcode_to_ptcode = {}
        self.llt_to_lltcode = {}
        self.lltcode_to_llt = {}
        self.ptcode_to_pt = {}
        self.pt_to_ptcode = {}
        self.ptcode_to_hltcode = {}
        self.hltcode_to_hlt = {}
        self.hlt_to_hltcode = {}
        self.ptcode_to_hltcode = {}
        self.hlgtcode_to_hlgt = {}
        self.hlgt_to_hlgtcode = {}
        self.hltcode_to_hlgtcode = {}
        self.soccode_to_soc = {}
        self.soc_to_soccode = {}
        self.hlgtcode_to_soccode = {}
        
        #если был дан umls - парсим и umls
        if umls_path:
            self.umls_path = umls_path
            self.umlscode_to_ptcode = {}
            self.umlscode_to_pt = {}
            self.umlscode_to_lltcode = {}
            self.umlscode_to_llt = {}
            self.umlscode_to_hltcode = {}
            self.umlscode_to_hlt = {}
            self.umlscode_to_hlgtcode = {}
            self.umlscode_to_hlgt = {}
            self.umlscode_to_soccode = {}
            self.umlscode_to_soc = {}
        else:
            self.umls_path = None

        self.Load()
        self.CreateMeddraTree()

    def Load(self):
        """
        собираем из медры только то, что надо, а именно:
        - все пт коды для быстрой проверки, является ли найденный где-то код PT кодом
        """
        with open(self.meddra_path_llt, "r", encoding='utf-8') as f:
            for llt_line in f:
                llt_line = llt_line.split("$")
                self.lltcodes.append(llt_line[0])
                self.lltcode_to_ptcode[llt_line[0]] = llt_line[2]
                self.llt_to_lltcode[llt_line[1]] = llt_line[0]
                
        with open(self.meddra_path_pt, "r", encoding='utf-8') as f:
            for pt_line in f:
                pt_line = pt_line.split("$")
                self.ptcodes.append(pt_line[0])
                self.ptcode_to_pt[pt_line[0]] = pt_line[1]
                self.pt_to_ptcode[pt_line[1]] = pt_line[0]
                
        with open(self.meddra_path_hlt, "r", encoding='utf-8') as f, open(self.meddra_path_hlt_to_pt, "r", encoding='utf-8') as map_f:
            for hlt_line in f:
                hlt_line = hlt_line.split("$")
                self.hltcodes.append(hlt_line[0])
                self.hlt_to_hltcode[hlt_line[1]] = hlt_line[0]
                self.hltcode_to_hlt[hlt_line[0]] = hlt_line[1]
            for map_line in map_f:
                map_line = map_line.split("$")
                self.ptcode_to_hltcode[map_line[1]] = map_line[0]
                
        with open(self.meddra_path_hlgt, "r", encoding='utf-8') as f, open(self.meddra_path_hlgt_to_hlt, "r", encoding='utf-8') as map_f:
            for hlgt_line in f:
                hlgt_line = hlgt_line.split("$")
                self.hlgtcodes.append(hlgt_line[0])
                self.hlgt_to_hlgtcode[hlgt_line[1]] = hlgt_line[0]
                self.hlgtcode_to_hlgt[hlgt_line[0]] = hlgt_line[1]
            for map_line in map_f:
                map_line = map_line.split("$")
                self.hltcode_to_hlgtcode[map_line[1]] = map_line[0]
                
        with open(self.meddra_path_soc, "r", encoding='utf-8') as f, open(self.meddra_path_soc_to_hlgt, "r", encoding='utf-8') as map_f:
            for soc_line in f:
                soc_line = soc_line.split("$")
                self.soccodes.append(soc_line[0])
                self.soc_to_soccode[soc_line[1]] = soc_line[0]
                self.soccode_to_soc[soc_line[0]] = soc_line[1]
            for map_line in map_f:
                map_line = map_line.split("$")
                self.hlgtcode_to_soccode[map_line[1]] = map_line[0]
        
        if self.umls_path:
            with open(self.umls_path, "r", encoding='utf-8') as f: #./Data/ExternalMRCONSO.RRF 12 и 13 строки
                for umls_line in f:
                    umls_line_values = umls_line.split('|')
                    umls_meddra_flag, umls_meddra_identity, lang_flag = umls_line_values[11], umls_line_values[12], umls_line_values[1] 
                    umls_meddra_flag = True if umls_meddra_flag=='MDR' else False
                    lang_flag = True if self.language == lang_flag else False
                    if not umls_meddra_flag or not lang_flag:
                        continue
                    attr_name_term_matching = 'umlscode_to_' + umls_meddra_identity.lower()
                    attr_name_code_matching = 'umlscode_to_' + umls_meddra_identity.lower() + 'code'
                    umls_code = umls_line_values[0]
                    meddra_code = umls_line_values[10] 
                    meddra_term = umls_line_values[14]
                    self.__dict__[attr_name_term_matching] = meddra_term
                    self.__dict__[attr_name_term_matching] = meddra_code
    #def CreateMeddraChainMap(self):
    #    self.MeddraCodesChainMap = ChainMap(self.lltcode_to_ptcode, self.ptcode_to_hltcode, self.hltcode_to_hlgtcode, self.hlgtcode_to_soccode)
        
    def RecursiveDescent(self, DictMapChain, nodes):
        if DictMapChain == []:
            return
        CurrMapDict = DictMapChain[-1]
        children_depth_level = CurrMapDict.split('_')[0]
        for node in nodes:
            another_node_children = [Node(k, depth_level = children_depth_level) \
                                     for k in self.__dict__[CurrMapDict].keys() if self.__dict__[CurrMapDict][k]==node.name]
            node.children = another_node_children
            return self.RecursiveDescent(DictMapChain[:-1], another_node_children)
        
    def CreateTreeList(self, DictMapChain, RelevantCodesSet = None):
        #собираем список всех корневых кодов, будем строить от них деревья в цикле
        root_nodes = []
        CurrMapDict = DictMapChain[-1]
        root_depth_level = CurrMapDict.split('_')[2]
        children_depth_level = CurrMapDict.split('_')[0]
        if RelevantCodesSet is None:
            RelevantCodesSet = set(self.__dict__[CurrMapDict].values())
        for root_node in tqdm(RelevantCodesSet):
            #собираем все ключи соответствующие корневому коду MedDRA
            node_children = [Node(k, depth_level = children_depth_level) for k in self.__dict__[CurrMapDict].keys() \
                             if self.__dict__[CurrMapDict][k]==root_node]
            root_node = Node(root_node, depth_level=root_depth_level)
            root_node.children = node_children
            root_nodes.append(root_node)
            self.RecursiveDescent(DictMapChain[:-1], node_children)
        return root_nodes
    
    def GetTree(self, code, level=None):
        #возвращает node кода с заданным уровнем (если level задан) c самым длинным путем, что гарантирует полный охват от нижнего кода до верхнего
        assert level in [None, 'llt', 'pt', 'hlt']
        result_nodes_varients = []
        for root_depth_level in self.root_codes_depth_levels:
            for root_node in self.__dict__[root_depth_level + '_root_nodes']:
                res = findall_by_attr(root_node, code)
                if res:
                    result_nodes_varients.extend(list(res))
        if len(result_nodes_varients)!=0:
            #assert level, '''There are several search results, please choose one
            #of the MedDRA levels (llt, pt, hlt) to determine the level on which your code %s exists'''%codes
            if level:
                candidates_on_level_for_code = []
                for node_varient in result_nodes_varients:
                    if node_varient.depth_level == level+'code':
                        candidates_on_level_for_code.append(node_varient)
                if candidates_on_level_for_code:
                    return max(candidates_on_level_for_code, key=lambda x: len(x.path))
                else:
                    return None #'No code %s on the level %s'%(code, level)
            else:
                return max(result_nodes_varients, key=lambda x: len(x.path))
        else:
            return None
       
        
    def CreateMeddraTree(self):
        #Парсит всю MedDRA, записывая корневые коды MedDRA в атрибуты класса с приставкой _root_nodes
        FullDictMapChain = ['ptcode_to_hltcode', 'hltcode_to_hlgtcode', 'hlgtcode_to_soccode'] #'lltcode_to_ptcode', 
        self.root_codes_depth_levels = []
        #начнем строить деревья с soc кодов до llt кодов
        DictMapChain = FullDictMapChain
        CurrMapDict = FullDictMapChain[-1]
        root_depth_level = 'soccode'
        self.root_codes_depth_levels.append(root_depth_level)
        print('Parsing %ss, creating trees'%root_depth_level)
        setattr(self, root_depth_level+'_root_nodes', self.CreateTreeList(DictMapChain))
        #На каждой новой итерации цикла добавляются деревья от кодов, которые не попали в предыдущие деревья
        for i in range(1, 3): #цифра 3, потому что нас интересуют все коды до pt, pt коды встречаются в 3ем MapDict от конца
            DictMapChain = FullDictMapChain[:-i]
            CurrMapDict = FullDictMapChain[:-i][-1]
            remaining_codes_set = set()
            curr_depth_level = CurrMapDict.split('_')[2]
            for curr_code in self.__dict__[curr_depth_level+'s']: #currcodes
                if not self.GetTree(curr_code):
                    remaining_codes_set.add(curr_code)
            if remaining_codes_set:
                print('There are %s codes in %s, which are not in current Trees list'%(len(remaining_codes_set), curr_depth_level))
                self.root_codes_depth_levels.append(curr_depth_level)
                print('Parsing %ss, creating trees'%curr_depth_level)
                setattr(self, curr_depth_level+'_root_nodes', self.CreateTreeList(DictMapChain, RelevantCodesSet=remaining_codes_set))
        #Таким образом предыдущим циклом мы охватили все pt коды
    
    def WalkUpper(self, code, level=None):
        #возвращает путь от нижнего кода code к верхнему коду soc
        try:
            soccode_node, code_node = self.GetTree(code, level)
        except:
            return None
        all_path_nodes = code_node.path
        all_path_codes = [node.name for node in all_path_nodes]
        all_path_codes.reverse()
        return all_path_codes
        
    def force_umlscode_to_ptcode(self, umls_code):
        #если код umls не приводится сразу к pt
        #тогда пробуем привести к llt, затем к pt
        if umls_code in self.umlscode_to_ptcode.keys():
            return self.umlscode_to_ptcode[umls_code]
        elif umls_code in self.umlscode_to_lltcode.keys():
            return self.lltcode_to_ptcode[self.umlscode_to_lltcode[umls_code]]
        else:
            return None