 // 定义图中的节点，具有 ID、类型和可选属性
class SimpleNode {  
  id string                   // 节点的唯一标识符
  type string                // 节点的类型/类别
  properties Properties      // 与节点关联的附加属性
}  

// 定义节点或关系的可选属性结构
class Properties {  
  description string?        // 可选的文本描述
}  

// 定义两个节点之间的关系
class SimpleRelationship {  
  source_node_id string      // 源节点的 ID
  source_node_type string    // 源节点的类型
  target_node_id string      // 目标节点的 ID
  target_node_type string    // 目标节点的类型
  type string                // 关系类型（例如，"connects_to"、"belongs_to"）
  properties Properties      // 关系的附加属性
}  

// 定义由节点和关系组成的整体图
class DynamicGraph {  
  nodes SimpleNode[]               // 图中所有节点的列表
  relationships SimpleRelationship[] // 节点之间所有关系的列表
}  

// 从原始输入字符串中提取 DynamicGraph 的函数
function ExtractGraph(graph: string) -> DynamicGraph {  
  client Ollama                   // 使用 Ollama 客户端解释输入
  prompt #"  
    Extract from this content:  
    {{ ctx.output_format }}  

    {{ graph }}                           // 提示模板指示 Ollama 提取图
 }