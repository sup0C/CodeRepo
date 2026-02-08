from question_classifier import *
from question_parser import *
from answer_search import *

'''问答类'''
class ChatBotGraph:
    def __init__(self):
        self.classifier = QuestionClassifier() # 构建分类树模型
        self.parser = QuestionPaser() #
        self.searcher = AnswerSearcher()

    def chat_main(self, sent):
        answer = '您好，我是医药智能助理，希望可以帮到您。如果没答上来，可以联系xxx。祝您身体棒棒！'
        res_classify = self.classifier.classify(sent) # 返回从问题中匹配到的用户关心、询问的点
        if not res_classify:
            return answer
        res_sql = self.parser.parser_main(res_classify) # list。里面有N个词典，每个词典有两个元素，分别是question_type和目标执行的N条cql语句
                                                        # 如：[{'question_type': 'disease_symptom', 'sql': ["MATCH (m:Disease)-[r:has_symptom]->(n:Symptom) where m.name = '感冒' return m.name, r.name, n.name"]}, {'question_type': 'disease_not_food', 'sql': ["MATCH (m:Disease)-[r:no_eat]->(n:Food) where m.name = '感冒' return m.name, r.name, n.name"]}, {'question_type': 'disease_drug', 'sql': ["MATCH (m:Disease)-[r:common_drug]->(n:Drug) where m.name = '感冒' return m.name, r.name, n.name", "MATCH (m:Disease)-[r:recommand_drug]->(n:Drug) where m.name = '感冒' return m.name, r.name, n.name"]}, {'question_type': 'disease_lasttime', 'sql': ["MATCH (m:Disease) where m.name = '感冒' return m.name, m.cure_lasttime"]}]
        final_answers = self.searcher.search_main(res_sql)
        if not final_answers:
            return answer
        else:
            return '\n'.join(final_answers)

if __name__ == '__main__':
    handler = ChatBotGraph()
    while 1:
        question = input('咨询:')
        answer = handler.chat_main(question)
        print('客服机器人:', answer)

