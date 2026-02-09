class QuestionPaser:
    '''构建实体字典：将分类器识别出的实体按类型分组'''

    def build_entitydict(self, args):
        """
        将输入的实体及其类型转换为按类型组织的字典。
        Args:
            args (dict): 例如 {'感冒': ['disease'], '头痛': ['symptom']}
        Returns:
            dict: 按类型分组的实体字典，例如 {'disease': ['感冒'], 'symptom': ['头痛']}
        """
        entity_dict = {}
        for arg, types in args.items():
            for type in types:
                if type not in entity_dict:
                    # 如果该类型还未在字典中，创建一个新列表
                    entity_dict[type] = [arg]
                else:
                    # 否则追加到已有列表中
                    entity_dict[type].append(arg)
        return entity_dict

    '''解析主函数：根据问题类型生成对应的数据库查询语句'''

    def parser_main(self, res_classify):
        """
        主要解析函数，接收分类结果并生成相应的 Cypher 查询语句。
        Args:
            res_classify (dict): 分类器输出的结果，包含 args 和 question_types
        Returns:
            list: 包含多个查询结构的列表，每个元素是 {'question_type': str, 'sql': list}
        """
        # 提取识别出的实体
        args = res_classify['args']
        # 构建按类型分组的实体字典
        entity_dict = self.build_entitydict(args)
        # 获取问题类型列表（如 ['disease_symptom']）
        question_types = res_classify['question_types']

        # 存储最终生成的所有 SQL（Cypher）语句
        sqls = []

        # 遍历每一种问题类型
        for question_type in question_types:
            # 初始化当前查询结构
            sql_ = {}
            sql_['question_type'] = question_type
            sql = []  # 当前类型的查询语句列表

            # 根据不同的问题类型调用 sql_transfer 方法生成查询语句
            if question_type == 'disease_symptom':
                # 疾病 → 症状
                sql = self.sql_transfer(question_type, entity_dict.get('disease'))

            elif question_type == 'symptom_disease':
                # 症状 → 可能的疾病
                sql = self.sql_transfer(question_type, entity_dict.get('symptom'))

            elif question_type == 'disease_cause':
                # 疾病 → 原因
                sql = self.sql_transfer(question_type, entity_dict.get('disease'))

            elif question_type == 'disease_acompany':
                # 疾病 → 并发症
                sql = self.sql_transfer(question_type, entity_dict.get('disease'))

            elif question_type == 'disease_not_food':
                # 疾病 → 忌吃食物
                sql = self.sql_transfer(question_type, entity_dict.get('disease'))

            elif question_type == 'disease_do_food':
                # 疾病 → 宜吃/推荐食物
                sql = self.sql_transfer(question_type, entity_dict.get('disease'))

            elif question_type == 'food_not_disease':
                # 食物 → 哪些疾病不能吃它
                sql = self.sql_transfer(question_type, entity_dict.get('food'))

            elif question_type == 'food_do_disease':
                # 食物 → 哪些疾病推荐吃它
                sql = self.sql_transfer(question_type, entity_dict.get('food'))

            elif question_type == 'disease_drug':
                # 疾病 → 推荐药品
                sql = self.sql_transfer(question_type, entity_dict.get('disease'))

            elif question_type == 'drug_disease':
                # 药品 → 可治疗的疾病
                sql = self.sql_transfer(question_type, entity_dict.get('drug'))

            elif question_type == 'disease_check':
                # 疾病 → 需要做哪些检查
                sql = self.sql_transfer(question_type, entity_dict.get('disease'))

            elif question_type == 'check_disease':
                # 检查项目 → 可能用于诊断哪些疾病
                sql = self.sql_transfer(question_type, entity_dict.get('check'))

            elif question_type == 'disease_prevent':
                # 疾病 → 如何预防
                sql = self.sql_transfer(question_type, entity_dict.get('disease'))

            elif question_type == 'disease_lasttime':
                # 疾病 → 治疗周期/持续时间
                sql = self.sql_transfer(question_type, entity_dict.get('disease'))

            elif question_type == 'disease_cureway':
                # 疾病 → 治疗方式
                sql = self.sql_transfer(question_type, entity_dict.get('disease'))

            elif question_type == 'disease_cureprob':
                # 疾病 → 治愈概率
                sql = self.sql_transfer(question_type, entity_dict.get('disease'))

            elif question_type == 'disease_easyget':
                # 疾病 → 易感人群
                sql = self.sql_transfer(question_type, entity_dict.get('disease'))

            elif question_type == 'disease_desc':
                # 疾病 → 基本介绍/描述
                sql = self.sql_transfer(question_type, entity_dict.get('disease'))

            # 如果成功生成了查询语句，则加入最终结果
            if sql:
                sql_['sql'] = sql
                sqls.append(sql_)

        return sqls

    '''根据问题类型和实体生成具体的 Cypher 查询语句'''

    def sql_transfer(self, question_type, entities):
        """
        核心方法：根据问题类型和相关实体生成 Neo4j 图数据库的查询语句（Cypher）
        Args:
            question_type (str): 问题类型，如 'disease_symptom'
            entities (list or None): 实体名称列表，如 ['感冒', '糖尿病']
        Returns:
            list: 生成的 Cypher 查询语句字符串列表
        """
        # 如果没有提取到有效实体，返回空列表
        if not entities:
            return []

        # 存储生成的查询语句
        sql = []

        # ================== 单向属性查询（节点自身属性）==================
        # 查询疾病的原因
        if question_type == 'disease_cause':
            sql = [
                "MATCH (m:Disease) WHERE m.name = '{0}' RETURN m.name, m.cause".format(i)
                for i in entities
            ]

        # 查询疾病的预防措施
        elif question_type == 'disease_prevent':
            sql = [
                "MATCH (m:Disease) WHERE m.name = '{0}' RETURN m.name, m.prevent".format(i)
                for i in entities
            ]

        # 查询疾病的持续时间（治疗周期）
        elif question_type == 'disease_lasttime':
            sql = [
                "MATCH (m:Disease) WHERE m.name = '{0}' RETURN m.name, m.cure_lasttime".format(i)
                for i in entities
            ]

        # 查询疾病的治愈概率
        elif question_type == 'disease_cureprob':
            sql = [
                "MATCH (m:Disease) WHERE m.name = '{0}' RETURN m.name, m.cured_prob".format(i)
                for i in entities
            ]

        # 查询疾病的治疗方式
        elif question_type == 'disease_cureway':
            sql = [
                "MATCH (m:Disease) WHERE m.name = '{0}' RETURN m.name, m.cure_way".format(i)
                for i in entities
            ]

        # 查询疾病的易感人群
        elif question_type == 'disease_easyget':
            sql = [
                "MATCH (m:Disease) WHERE m.name = '{0}' RETURN m.name, m.easy_get".format(i)
                for i in entities
            ]

        # 查询疾病的基本描述
        elif question_type == 'disease_desc':
            sql = [
                "MATCH (m:Disease) WHERE m.name = '{0}' RETURN m.name, m.desc".format(i)
                for i in entities
            ]

        # ================== 关系型查询（通过边查询其他节点）==================
        # 查询疾病有哪些症状
        elif question_type == 'disease_symptom':
            sql = [
                "MATCH (m:Disease)-[r:has_symptom]->(n:Symptom) WHERE m.name = '{0}' "
                "RETURN m.name, r.name, n.name".format(i)
                for i in entities
            ]

        # 查询某个症状可能对应哪些疾病（逆向关系）
        elif question_type == 'symptom_disease':
            sql = [
                "MATCH (m:Disease)-[r:has_symptom]->(n:Symptom) WHERE n.name = '{0}' "
                "RETURN m.name, r.name, n.name".format(i)
                for i in entities
            ]

        # 查询疾病的并发症（双向：该病引发的 + 会引发该病的）
        elif question_type == 'disease_acompany':
            # 1. 该病会并发哪些其他疾病
            sql1 = [
                "MATCH (m:Disease)-[r:acompany_with]->(n:Disease) WHERE m.name = '{0}' "
                "RETURN m.name, r.name, n.name".format(i)
                for i in entities
            ]
            # 2. 哪些疾病会并发此病（即它是别人的并发症）
            sql2 = [
                "MATCH (m:Disease)-[r:acompany_with]->(n:Disease) WHERE n.name = '{0}' "
                "RETURN m.name, r.name, n.name".format(i)
                for i in entities
            ]
            sql = sql1 + sql2  # 合并两个方向的查询

        # 查询疾病忌吃的食物
        elif question_type == 'disease_not_food':
            sql = [
                "MATCH (m:Disease)-[r:no_eat]->(n:Food) WHERE m.name = '{0}' "
                "RETURN m.name, r.name, n.name".format(i)
                for i in entities
            ]

        # 查询疾病宜吃或推荐的食物（两种关系合并）
        elif question_type == 'disease_do_food':
            sql1 = [
                "MATCH (m:Disease)-[r:do_eat]->(n:Food) WHERE m.name = '{0}' "
                "RETURN m.name, r.name, n.name".format(i)
                for i in entities
            ]
            sql2 = [
                "MATCH (m:Disease)-[r:recommand_eat]->(n:Food) WHERE m.name = '{0}' "
                "RETURN m.name, r.name, n.name".format(i)
                for i in entities
            ]
            sql = sql1 + sql2

        # 已知某种食物不能吃，反查哪些疾病需要忌口
        elif question_type == 'food_not_disease':
            sql = [
                "MATCH (m:Disease)-[r:no_eat]->(n:Food) WHERE n.name = '{0}' "
                "RETURN m.name, r.name, n.name".format(i)
                for i in entities
            ]

        # 已知某种食物推荐吃，反查对应哪些疾病
        elif question_type == 'food_do_disease':
            sql1 = [
                "MATCH (m:Disease)-[r:do_eat]->(n:Food) WHERE n.name = '{0}' "
                "RETURN m.name, r.name, n.name".format(i)
                for i in entities
            ]
            sql2 = [
                "MATCH (m:Disease)-[r:recommand_eat]->(n:Food) WHERE n.name = '{0}' "
                "RETURN m.name, r.name, n.name".format(i)
                for i in entities
            ]
            sql = sql1 + sql2

        # 查询疾病常用或推荐的药物
        elif question_type == 'disease_drug':
            sql1 = [
                "MATCH (m:Disease)-[r:common_drug]->(n:Drug) WHERE m.name = '{0}' "
                "RETURN m.name, r.name, n.name".format(i)
                for i in entities
            ]
            sql2 = [
                "MATCH (m:Disease)-[r:recommand_drug]->(n:Drug) WHERE m.name = '{0}' "
                "RETURN m.name, r.name, n.name".format(i)
                for i in entities
            ]
            sql = sql1 + sql2

        # 已知药品，查询其可治疗的疾病
        elif question_type == 'drug_disease':
            sql1 = [
                "MATCH (m:Disease)-[r:common_drug]->(n:Drug) WHERE n.name = '{0}' "
                "RETURN m.name, r.name, n.name".format(i)
                for i in entities
            ]
            sql2 = [
                "MATCH (m:Disease)-[r:recommand_drug]->(n:Drug) WHERE n.name = '{0}' "
                "RETURN m.name, r.name, n.name".format(i)
                for i in entities
            ]
            sql = sql1 + sql2

        # 查询疾病需要做哪些检查
        elif question_type == 'disease_check':
            sql = [
                "MATCH (m:Disease)-[r:need_check]->(n:Check) WHERE m.name = '{0}' "
                "RETURN m.name, r.name, n.name".format(i)
                for i in entities
            ]

        # 已知检查项目，查询可用于诊断哪些疾病
        elif question_type == 'check_disease':
            sql = [
                "MATCH (m:Disease)-[r:need_check]->(n:Check) WHERE n.name = '{0}' "
                "RETURN m.name, r.name, n.name".format(i)
                for i in entities
            ]
        print("neo4j查询语句：", sql)

        return sql


# ================== 测试入口 ==================
if __name__ == '__main__':
    # 创建 QuestionPaser 实例
    handler = QuestionPaser()

    # 示例：模拟一个分类器输出结果
    res_classify_example = {
        'args': {
            '感冒': ['disease'],
            '头痛': ['symptom']
        },
        'question_types': ['disease_symptom', 'disease_cause']
    }

    # 调用主解析函数
    result = handler.parser_main(res_classify_example)

    # 打印生成的查询语句
    for item in result:
        print(f"问题类型: {item['question_type']}")
        for s in item['sql']:
            print(f"  查询语句: {s}")
