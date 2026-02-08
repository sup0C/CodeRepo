from py2neo import Graph

class AnswerSearcher:
    def __init__(self):
        # self.g = Graph("http://localhost:7474", username="neo4j", password="tangyudiadid0")
        self.g= Graph("bolt://localhost:7687", auth=("neo4j", "123qwe!@"))
        self.num_limit = 20

    '''执行cypher查询，并返回相应结果'''
    def search_main(self, sqls):
        '''
        :param sqls:list。里面有N个词典，每个词典有两个元素，分别是question_type和目标执行的N条cql语句
                                                        # 如：[{'question_type': 'disease_symptom', 'sql': ["MATCH (m:Disease)-[r:has_symptom]->(n:Symptom) where m.name = '感冒' return m.name, r.name, n.name"]}, {'question_type': 'disease_not_food', 'sql': ["MATCH (m:Disease)-[r:no_eat]->(n:Food) where m.name = '感冒' return m.name, r.name, n.name"]}, {'question_type': 'disease_drug', 'sql': ["MATCH (m:Disease)-[r:common_drug]->(n:Drug) where m.name = '感冒' return m.name, r.name, n.name", "MATCH (m:Disease)-[r:recommand_drug]->(n:Drug) where m.name = '感冒' return m.name, r.name, n.name"]}, {'question_type': 'disease_lasttime', 'sql': ["MATCH (m:Disease) where m.name = '感冒' return m.name, m.cure_lasttime"]}]
        :return:问题的回答，对于问题中多个侧重关注的领域（如疾病对应的饮食、科室、药物等），只返回该领域中匹配到的前self.num_limit个描述
        '''
        final_answers = []
        for sql_ in sqls:
            question_type = sql_['question_type'] # 问题的焦点，如'disease_symptom'
            queries = sql_['sql'] # 问题的相应cql语句
            answers = []
            for query in queries:
                ress = self.g.run(query).data() # 执行cql语句进行查询，cql返回的是cql语句中return的结果，即各个实体的name,如{'m.name': '肺气肿', 'r.name': '症状', 'n.name': '桶状胸'}
                answers += ress
            final_answer = self.answer_prettify(question_type, answers)
            if final_answer:
                final_answers.append(final_answer)
        return final_answers

    def answer_prettify(self, question_type, answers):
        '''
        根据对应的qustion_type，调用相应的回复模板
        :param question_type:问题的焦点，如'disease_symptom'
        :param answers:回答，即cql语句中return的结果，即各个实体的name,如{'m.name': '肺气肿', 'r.name': '症状', 'n.name': '桶状胸'}
        :return:
        '''
        final_answer = []
        if not answers:
            return ''
        if question_type == 'disease_symptom':
            desc = [i['n.name'] for i in answers] # 相应的描述
            subject = answers[0]['m.name'] # 主体。即disease名称
            final_answer = '{0}的症状包括：{1}'.format(subject, '；'.join(list(set(desc))[:self.num_limit])) # 对于问题中多个侧重关注的领域（如疾病对应的饮食、科室、药物等），只返回该领域中匹配到的前self.num_limit个描述

        elif question_type == 'symptom_disease':
            desc = [i['m.name'] for i in answers]
            subject = answers[0]['n.name']
            final_answer = '症状{0}可能染上的疾病有：{1}'.format(subject, '；'.join(list(set(desc))[:self.num_limit]))

        elif question_type == 'disease_cause':
            desc = [i['m.cause'] for i in answers]
            subject = answers[0]['m.name']
            final_answer = '{0}可能的成因有：{1}'.format(subject, '；'.join(list(set(desc))[:self.num_limit]))

        elif question_type == 'disease_prevent':
            desc = [i['m.prevent'] for i in answers]
            subject = answers[0]['m.name']
            final_answer = '{0}的预防措施包括：{1}'.format(subject, '；'.join(list(set(desc))[:self.num_limit]))

        elif question_type == 'disease_lasttime':
            desc = [i['m.cure_lasttime'] for i in answers]
            subject = answers[0]['m.name']
            final_answer = '{0}治疗可能持续的周期为：{1}'.format(subject, '；'.join(list(set(desc))[:self.num_limit]))

        elif question_type == 'disease_cureway':
            desc = [';'.join(i['m.cure_way']) for i in answers]
            subject = answers[0]['m.name']
            final_answer = '{0}可以尝试如下治疗：{1}'.format(subject, '；'.join(list(set(desc))[:self.num_limit]))

        elif question_type == 'disease_cureprob':
            desc = [i['m.cured_prob'] for i in answers]
            subject = answers[0]['m.name']
            final_answer = '{0}治愈的概率为（仅供参考）：{1}'.format(subject, '；'.join(list(set(desc))[:self.num_limit]))

        elif question_type == 'disease_easyget':
            desc = [i['m.easy_get'] for i in answers]
            subject = answers[0]['m.name']

            final_answer = '{0}的易感人群包括：{1}'.format(subject, '；'.join(list(set(desc))[:self.num_limit]))

        elif question_type == 'disease_desc':
            desc = [i['m.desc'] for i in answers]
            subject = answers[0]['m.name']
            final_answer = '{0},熟悉一下：{1}'.format(subject,  '；'.join(list(set(desc))[:self.num_limit]))

        elif question_type == 'disease_acompany':
            desc1 = [i['n.name'] for i in answers]
            desc2 = [i['m.name'] for i in answers]
            subject = answers[0]['m.name']
            desc = [i for i in desc1 + desc2 if i != subject]
            final_answer = '{0}的症状包括：{1}'.format(subject, '；'.join(list(set(desc))[:self.num_limit]))

        elif question_type == 'disease_not_food':
            desc = [i['n.name'] for i in answers]
            subject = answers[0]['m.name']
            final_answer = '{0}忌食的食物包括有：{1}'.format(subject, '；'.join(list(set(desc))[:self.num_limit]))

        elif question_type == 'disease_do_food':
            do_desc = [i['n.name'] for i in answers if i['r.name'] == '宜吃']
            recommand_desc = [i['n.name'] for i in answers if i['r.name'] == '推荐食谱']
            subject = answers[0]['m.name']
            final_answer = '{0}宜食的食物包括有：{1}\n推荐食谱包括有：{2}'.format(subject, ';'.join(list(set(do_desc))[:self.num_limit]), ';'.join(list(set(recommand_desc))[:self.num_limit]))

        elif question_type == 'food_not_disease':
            desc = [i['m.name'] for i in answers]
            subject = answers[0]['n.name']
            final_answer = '患有{0}的人最好不要吃{1}'.format('；'.join(list(set(desc))[:self.num_limit]), subject)

        elif question_type == 'food_do_disease':
            desc = [i['m.name'] for i in answers]
            subject = answers[0]['n.name']
            final_answer = '患有{0}的人建议多试试{1}'.format('；'.join(list(set(desc))[:self.num_limit]), subject)

        elif question_type == 'disease_drug':
            desc = [i['n.name'] for i in answers]
            subject = answers[0]['m.name']
            final_answer = '{0}通常的使用的药品包括：{1}'.format(subject, '；'.join(list(set(desc))[:self.num_limit]))

        elif question_type == 'drug_disease':
            desc = [i['m.name'] for i in answers]
            subject = answers[0]['n.name']
            final_answer = '{0}主治的疾病有{1},可以试试'.format(subject, '；'.join(list(set(desc))[:self.num_limit]))

        elif question_type == 'disease_check':
            desc = [i['n.name'] for i in answers]
            subject = answers[0]['m.name']
            final_answer = '{0}通常可以通过以下方式检查出来：{1}'.format(subject, '；'.join(list(set(desc))[:self.num_limit]))

        elif question_type == 'check_disease':
            desc = [i['m.name'] for i in answers]
            subject = answers[0]['n.name']
            final_answer = '通常可以通过{0}检查出来的疾病有{1}'.format(subject, '；'.join(list(set(desc))[:self.num_limit]))

        return final_answer


if __name__ == '__main__':
    searcher = AnswerSearcher()