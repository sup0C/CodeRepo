import os
import json
from py2neo import Graph, Node


class MedicalGraph:
    def __init__(self):
        cur_dir = '/'.join(os.path.abspath(__file__).split('/')[:-1])
        self.data_path = os.path.join(cur_dir, 'data/medical2.json')
        self.g = Graph(
            host="127.0.0.1",  # neo4j 搭载服务器的ip地址，ifconfig可获取到
            # http_port=7474,  # neo4j 服务器监听的端口号
            user="neo4j",  # 数据库user name，如果没有更改过，应该是neo4j
            password="123qwe!@")
        print("知识图谱连接成功！")

    '''读取文件'''

    def read_nodes(self):
        """
        从 JSON 数据文件中读取疾病数据，提取节点（实体）和关系，用于构建医疗知识图谱。
        返回所有去重后的节点集合及各类关系列表。
        """

        # ================== 初始化各类节点列表（实体） ==================
        drugs = []  # 存储药品名称
        foods = []  # 存储食物名称（宜吃、忌吃、推荐吃）
        checks = []  # 存储检查项目名称
        departments = []  # 存储科室名称（如内科、呼吸内科）
        producers = []  # 存储药品生产厂家
        diseases = []  # 存储疾病名称
        symptoms = []  # 存储症状名称

        disease_infos = []  # 存储每个疾病的详细信息字典

        # ================== 初始化实体间的关系列表 ==================
        rels_department = []  # 科室-上级科室关系（如：呼吸内科 -> 内科）
        rels_noteat = []  # 疾病-忌吃食物关系
        rels_doeat = []  # 疾病-宜吃食物关系
        rels_recommandeat = []  # 疾病-推荐吃食物关系
        rels_commonddrug = []  # 疾病-常用药品关系
        rels_recommanddrug = []  # 疾病-推荐药品关系
        rels_check = []  # 疾病-所需检查项目关系
        rels_drug_producer = []  # 药品-生产厂家关系

        rels_symptom = []  # 疾病-症状关系
        rels_acompany = []  # 疾病-并发疾病关系（并发症）
        rels_category = []  # 疾病-所属科室关系

        count = 0
        print("开始构建知识图谱...")
        '''
        这是第一条数据 data[0]
        {
            "_id": {"$oid": "5bb578b6831b973a137e3ee6"},
            "name": "肺泡蛋白质沉积症",
            "desc": "肺泡蛋白质沉积症(简称PAP)，又称Rosen-Castleman-Liebow综合征，是一种罕见疾病。
            该病以肺泡和细支气管腔内充满PAS染色阳性，来自肺的富磷脂蛋白质物质为其特征，好发于青中年，男性发病约3倍于女性。",
            "category": ["疾病百科", "内科", "呼吸内科"],
            "prevent": "1、避免感染分支杆菌病，卡氏肺囊肿肺炎，巨细胞病毒等。2、注意锻炼身体，提高免疫力。",
            "cause": "病因未明，推测与几方面因素有关：如大量粉尘吸入（铝，二氧化硅等），机体免疫功能下降（尤其婴幼儿），
            遗传因素，酗酒，微生物感染等。对于感染，有时很难确认是原发致病因素还是继发于肺泡蛋白沉着症，
            例如巨细胞病毒、卡氏肺孢子虫、组织胞浆菌感染等均发现有肺泡内高蛋白沉着。虽然启动因素尚不明确，
            但基本上同意发病过程为脂质代谢障碍所致，即由于机体内、外因素作用引起肺泡表面活性物质的代谢异常。
            目前研究较多的是肺泡巨噬细胞活力。动物实验证明巨噬细胞吞噬粉尘后其活力明显下降，而患者灌洗液中的巨噬细胞内颗粒可使正常细胞活力下降。
            经支气管肺泡灌洗治疗后，其肺泡巨噬细胞活力可上升。研究未发现Ⅱ型肺泡上皮细胞生成蛋白增加，全身脂代谢也无异常。因此目前一般认为本病与清除能力下降有关。",
            "symptom": ["紫绀", "胸痛", "呼吸困难", "乏力", "咳嗽"],
            "yibao_status": "否",
            "get_prob": "0.00002%",
            "get_way": "无传染性",
            "acompany": ["多重肺部感染"],
            "cure_department": ["内科", "呼吸内科"],
            "cure_way": ["支气管肺泡灌洗"],
            "cure_lasttime": "约3个月",
            "cured_prob": "约40%",
            "cost_money": "根据不同医院，收费标准不一致，省市三甲医院约（8000——15000元）",
            "check": ["胸部CT检查", "肺活检", "支气管镜检查"],
            "recommand_drug": [],
            "drug_detail": []
        }
        '''

        # ================== 遍历数据文件中的每一行（每条疾病数据） ==================
        for data in open(self.data_path, 'r'):
            disease_dict = {}  # 当前疾病的详细信息字典
            count += 1
            print(count)
            data_json = json.loads(data)  # 将 JSON 字符串解析为 Python 字典
            disease = data_json['name']
            disease_dict['name'] = disease  # 记录疾病名称
            diseases.append(disease)

            # 初始化疾病详细信息字段
            disease_dict['desc'] = ''
            disease_dict['prevent'] = ''
            disease_dict['cause'] = ''
            disease_dict['easy_get'] = ''
            disease_dict['cure_department'] = ''
            disease_dict['cure_way'] = ''
            disease_dict['cure_lasttime'] = ''
            disease_dict['symptom'] = ''
            disease_dict['cured_prob'] = ''

            # -------------------- 模块1: 处理症状关系 --------------------
            if 'symptom' in data_json:
                # 将当前疾病的所有症状加入全局症状列表
                symptoms += data_json['symptom']
                # 构建“疾病-症状”关系对
                for symptom in data_json['symptom']:
                    rels_symptom.append([disease, symptom])

            # -------------------- 模块2: 处理并发症关系 --------------------
            if 'acompany' in data_json:
                # 构建“疾病-并发疾病”关系对
                for acompany in data_json['acompany']:
                    rels_acompany.append([disease, acompany])

            # -------------------- 模块3: 处理疾病描述 --------------------
            if 'desc' in data_json:
                disease_dict['desc'] = data_json['desc']

            # -------------------- 模块4: 处理预防措施 --------------------
            if 'prevent' in data_json:
                disease_dict['prevent'] = data_json['prevent']

            # -------------------- 模块5: 处理病因 --------------------
            if 'cause' in data_json:
                disease_dict['cause'] = data_json['cause']

            # -------------------- 模块6: 处理发病率 --------------------
            if 'get_prob' in data_json:
                disease_dict['get_prob'] = data_json['get_prob']

            # -------------------- 模块7: 处理易感人群（注意：字段名可能应为 easy_get，但数据中可能未使用） --------------------
            if 'easy_get' in data_json:
                disease_dict['easy_get'] = data_json['easy_get']

            # -------------------- 模块8: 处理就诊科室 --------------------
            if 'cure_department' in data_json:
                cure_department = data_json['cure_department']
                disease_dict['cure_department'] = cure_department
                departments += cure_department  # 添加到全局科室列表

                # 根据科室层级构建关系
                if len(cure_department) == 1:
                    # 只有一个科室：直接建立疾病与科室的关系
                    rels_category.append([disease, cure_department[0]])
                elif len(cure_department) == 2:
                    # 有两个科室：认为第二个是子科室，第一个是父科室
                    big = cure_department[0]  # 父科室（如：内科）
                    small = cure_department[1]  # 子科室（如：呼吸内科）
                    rels_department.append([small, big])  # 子科室 → 父科室
                    rels_category.append([disease, small])  # 疾病 → 子科室

            # -------------------- 模块9: 处理治疗方式 --------------------
            if 'cure_way' in data_json:
                disease_dict['cure_way'] = data_json['cure_way']

            # -------------------- 模块10: 处理治疗周期 --------------------
            if 'cure_lasttime' in data_json:
                disease_dict['cure_lasttime'] = data_json['cure_lasttime']

            # -------------------- 模块11: 处理治愈概率 --------------------
            if 'cured_prob' in data_json:
                disease_dict['cured_prob'] = data_json['cured_prob']

            # -------------------- 模块12: 处理常用药品 --------------------
            if 'common_drug' in data_json:
                common_drug = data_json['common_drug']
                drugs += common_drug  # 添加到全局药品列表
                # 构建“疾病-常用药品”关系
                for drug in common_drug:
                    rels_commonddrug.append([disease, drug])

            # -------------------- 模块13: 处理推荐药品 --------------------
            if 'recommand_drug' in data_json:
                recommand_drug = data_json['recommand_drug']
                drugs += recommand_drug  # 添加到全局药品列表
                # 构建“疾病-推荐药品”关系
                for drug in recommand_drug:
                    rels_recommanddrug.append([disease, drug])

            # -------------------- 模块14: 处理饮食禁忌与推荐 --------------------
            if 'not_eat' in data_json:
                not_eat = data_json['not_eat']
                do_eat = data_json['do_eat']
                recommand_eat = data_json['recommand_eat']

                # 构建“疾病-忌吃食物”关系
                for _not in not_eat:
                    rels_noteat.append([disease, _not])
                foods += not_eat

                # 构建“疾病-宜吃食物”关系
                for _do in do_eat:
                    rels_doeat.append([disease, _do])
                foods += do_eat

                # 构建“疾病-推荐吃食物”关系
                for _recommand in recommand_eat:
                    rels_recommandeat.append([disease, _recommand])
                foods += recommand_eat

            # -------------------- 模块15: 处理所需检查 --------------------
            if 'check' in data_json:
                check = data_json['check']
                checks += check  # 添加到全局检查项目列表
                # 构建“疾病-检查项目”关系
                for _check in check:
                    rels_check.append([disease, _check])

            # -------------------- 模块16: 处理药品详情（含厂家） --------------------
            if 'drug_detail' in data_json:
                drug_detail = data_json['drug_detail']
                # 提取药品名称（括号前部分）
                producer = [i.split('(')[0] for i in drug_detail]
                # 提取“药品-厂家”关系对
                rels_drug_producer += [[i.split('(')[0], i.split('(')[-1].replace(')', '')] for i in drug_detail]
                producers += producer  # 添加到全局厂家列表
                drugs += producer  # 药品名称也加入药品列表

            # -------------------- 汇总当前疾病信息 --------------------
            disease_infos.append(disease_dict)

        # ================== 返回所有去重后的节点和关系列表 ==================
        return (
            set(drugs),  # 去重药品
            set(foods),  # 去重食物
            set(checks),  # 去重检查
            set(departments),  # 去重科室
            set(producers),  # 去重药厂
            set(symptoms),  # 去重症状
            set(diseases),  # 去重疾病
            disease_infos,  # 疾病详细信息列表
            # 关系列表
            rels_check,
            rels_recommandeat,
            rels_noteat,
            rels_doeat,
            rels_department,
            rels_commonddrug,
            rels_drug_producer,
            rels_recommanddrug,
            rels_symptom,
            rels_acompany,
            rels_category
        )

    '''建立节点'''

    def create_node(self, label, nodes):
        count = 0
        for node_name in nodes:
            node = Node(label, name=node_name)
            self.g.create(node)
            count += 1
            print(count, len(nodes))
        return

    '''创建知识图谱中心疾病的节点，添加属性'''

    def create_diseases_nodes(self, disease_infos):
        count = 0
        for disease_dict in disease_infos:
            node = Node("Disease", name=disease_dict['name'], desc=disease_dict['desc'],
                        prevent=disease_dict['prevent'], cause=disease_dict['cause'],
                        easy_get=disease_dict['easy_get'], cure_lasttime=disease_dict['cure_lasttime'],
                        cure_department=disease_dict['cure_department']
                        , cure_way=disease_dict['cure_way'], cured_prob=disease_dict['cured_prob'])
            self.g.create(node)
            count += 1
            print(count)
        return

    '''创建知识图谱实体节点类型schema'''

    def create_graphnodes(self):
        (Drugs, Foods, Checks, Departments, Producers, Symptoms, Diseases, disease_infos, rels_check,
         rels_recommandeat, rels_noteat, rels_doeat, rels_department, rels_commonddrug,
         rels_drug_producer, rels_recommanddrug, rels_symptom, rels_acompany, rels_category) = self.read_nodes()
        self.create_diseases_nodes(disease_infos)
        self.create_node('Drug', Drugs)
        print(len(Drugs))
        self.create_node('Food', Foods)
        print(len(Foods))
        self.create_node('Check', Checks)
        print(len(Checks))
        self.create_node('Department', Departments)
        print(len(Departments))
        self.create_node('Producer', Producers)
        print(len(Producers))
        self.create_node('Symptom', Symptoms)
        return

    '''创建实体关系边'''

    def create_graphrels(self):
        Drugs, Foods, Checks, Departments, Producers, Symptoms, Diseases, disease_infos, rels_check, rels_recommandeat, rels_noteat, rels_doeat, rels_department, rels_commonddrug, rels_drug_producer, rels_recommanddrug, rels_symptom, rels_acompany, rels_category = self.read_nodes()
        self.create_relationship('Disease', 'Food', rels_recommandeat, 'recommand_eat', '推荐食谱')
        self.create_relationship('Disease', 'Food', rels_noteat, 'no_eat', '忌吃')
        self.create_relationship('Disease', 'Food', rels_doeat, 'do_eat', '宜吃')
        self.create_relationship('Department', 'Department', rels_department, 'belongs_to', '属于')
        self.create_relationship('Disease', 'Drug', rels_commonddrug, 'common_drug', '常用药品')
        self.create_relationship('Producer', 'Drug', rels_drug_producer, 'drugs_of', '生产药品')
        self.create_relationship('Disease', 'Drug', rels_recommanddrug, 'recommand_drug', '好评药品')
        self.create_relationship('Disease', 'Check', rels_check, 'need_check', '诊断检查')
        self.create_relationship('Disease', 'Symptom', rels_symptom, 'has_symptom', '症状')
        self.create_relationship('Disease', 'Disease', rels_acompany, 'acompany_with', '并发症')
        self.create_relationship('Disease', 'Department', rels_category, 'belongs_to', '所属科室')

    '''创建实体关联边'''

    def create_relationship(self, start_node, end_node, edges, rel_type, rel_name):
        count = 0
        # 去重处理
        set_edges = []
        for edge in edges:
            set_edges.append('###'.join(edge))
        all = len(set(set_edges))
        for edge in set(set_edges):
            edge = edge.split('###')
            p = edge[0]
            q = edge[1]
            query = "match(p:%s),(q:%s) where p.name='%s'and q.name='%s' create (p)-[rel:%s{name:'%s'}]->(q)" % (
                start_node, end_node, p, q, rel_type, rel_name)
            try:
                self.g.run(query)
                count += 1
                print(rel_type, count, all)
            except Exception as e:
                print(e)
        return


    def export_data(self):
        '''
        导出数据：将数据导出为txt文件
        :return:
        '''
        Drugs, Foods, Checks, Departments, Producers, Symptoms, Diseases, disease_infos, rels_check, rels_recommandeat, rels_noteat, rels_doeat, rels_department, rels_commonddrug, rels_drug_producer, rels_recommanddrug, rels_symptom, rels_acompany, rels_category = self.read_nodes()
        f_drug = open('drug.txt', 'w+', encoding='utf-8')
        f_food = open('food.txt', 'w+', encoding='utf-8')
        f_check = open('check.txt', 'w+', encoding='utf-8')
        f_department = open('department.txt', 'w+', encoding='utf-8')
        f_producer = open('producer.txt', 'w+', encoding='utf-8')
        f_symptom = open('symptoms.txt', 'w+', encoding='utf-8')
        f_disease = open('disease.txt', 'w+', encoding='utf-8')

        f_drug.write('\n'.join(list(Drugs)))
        f_food.write('\n'.join(list(Foods)))
        f_check.write('\n'.join(list(Checks)))
        f_department.write('\n'.join(list(Departments)))
        f_producer.write('\n'.join(list(Producers)))
        f_symptom.write('\n'.join(list(Symptoms)))
        f_disease.write('\n'.join(list(Diseases)))

        f_drug.close()
        f_food.close()
        f_check.close()
        f_department.close()
        f_producer.close()
        f_symptom.close()
        f_disease.close()

        return


if __name__ == '__main__':
    handler = MedicalGraph()
    print("step1:导入图谱节点中")
    handler.create_graphnodes()
    print("step2:导入图谱边中")
    handler.create_graphrels()

    print("构建知识图谱完成，可以直接使用Neo4j可视化工具进行可视化展示")
