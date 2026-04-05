import { metrics } from './metadata.js';
import * as keras from './keras.js';
import * as pytorch from './pytorch.js';
import * as onnx from './onnx.js';
import * as tf from './tf.js';
import * as tflite from './tflite.js';

const diffAnalyser = {};

/* -------------------- Graph primitives -------------------- */

diffAnalyser.Edge = class {
    constructor(from) {
        this.from = from;
        this.to = null; // will become an array lazily
    }

    addTo(node) {
        if (this.to === null) {
            this.to = [];
        }
        this.to.push(node);
    }
};

diffAnalyser.StartNode = class {
    constructor(outputEdges) {
        this._outputs = outputEdges;
    }

    addEdge(edgeName) {
        const output = {value: [{name: edgeName}]};
        this._outputs.push(output);
    }

    get outputs() {
        return this._outputs;
    }
};

export const DiffStatus = Object.freeze({
    SAME: 'SAME',
    DIFF: 'DIFF',
    REMOVED: 'REMOVED',
    ADDED: 'ADDED',
    UNKNOWN: 'UNKNOWN'
});

diffAnalyser.PropertyStatus = class {
    constructor(generalStatus = DiffStatus.UNKNOWN, tensorValueStatus = DiffStatus.UNKNOWN) {
        this._generalStatus = generalStatus;      // Status of the property comparison without considering tensor values
        this._tensorValueStatus = tensorValueStatus; // Status of the tensor value comparison, or UNKNOWN for non-tensors
    }

    get generalStatus() {
        return this._generalStatus;
    }

    get tensorValueStatus() {
        return this._tensorValueStatus;
    }

    setGeneralStatus(generalStatus) {
        this._generalStatus = generalStatus;
    }

    setTensorValueStatus(tensorValueStatus) {
        this._tensorValueStatus = tensorValueStatus;
    }
};

diffAnalyser.NodeInfo = class {
    constructor(nodeID, generalStatus) {
        this._nodeID = nodeID;
        this._generalStatus = generalStatus;
    }

    get nodeID() {
        return this._nodeID;
    }

    get generalStatus() {
        return this._generalStatus;
    }
};

diffAnalyser.NodeInfos = class {
    constructor(nodeInfos, modelIdx) {
        this._nodeInfos = nodeInfos;
        this._modelIdx = modelIdx;
    }

    getNodeInfo(node) {
        let nodeInfo = this._nodeInfos.get(node);
        // If nodeInfo is undefined, it means the node is not matched and removed in the other model.
        if (nodeInfo === undefined) {
            if (this._modelIdx === 1) {
                nodeInfo = new diffAnalyser.NodeInfo(null, DiffStatus.REMOVED);
            }
            else if (this._modelIdx === 2) {
                nodeInfo = new diffAnalyser.NodeInfo(null, DiffStatus.ADDED);
            }
        }
        return nodeInfo;
    }
};

diffAnalyser.ModelDifferences = class {
    constructor(modelDiffEntries) {
        this._nodeDiffs = new Map();
        this._model1NodeInfos = null;
        this._model2NodeInfos = null;

        const model1NodeInfos = new Map();
        const model2NodeInfos = new Map();
        for (const entry of modelDiffEntries) {
            const nodeInfo = new diffAnalyser.NodeInfo(entry.nodeID, entry.generalStatus);
            model1NodeInfos.set(entry.node1, nodeInfo);
            model2NodeInfos.set(entry.node2, nodeInfo);
            this._nodeDiffs.set(entry.nodeID, entry.nodeDifferences);
        }
        this._model1NodeInfos = new diffAnalyser.NodeInfos(model1NodeInfos, 1);
        this._model2NodeInfos = new diffAnalyser.NodeInfos(model2NodeInfos, 2);
    }

    get model1NodeInfos() {
        return this._model1NodeInfos;
    }

    get model2NodeInfos() {
        return this._model2NodeInfos;
    }

    nodeDiffs(ID) {
        return this._nodeDiffs.get(ID);
    }
};

/* -------------------- Model Nodes Diff Analyzer -------------------- */

diffAnalyser.ModelNodesDiffAnalyzer = class {
    // -----------------------------
    // Helpers & Type Guards
    // -----------------------------

    static _hasOwn(obj, key) {
        return Object.hasOwn ? Object.hasOwn(obj, key) : Object.prototype.hasOwnProperty.call(obj, key);
    }

    static _isTensorValue(value) {
        // Tensor has name, quantization and initializer
        return "name" in value && "initializer" in value;
    }

    static _isInputTensorValue(value) {
        // Input tensor doesn't have type
        return this._isTensorValue(value) && value.initializer === null;
    }

    static _isPrimitiveArray(argument) {
        const notPrimitiveArray = argument.value !== null 
            && Array.isArray(argument.value) && argument.value.length > 0 
            && argument.type === null;
        return !notPrimitiveArray;
    }

    static _isInputTensorArgument(argument) {
        if (this._isPrimitiveArray(argument) ) {
            return false;
        }
        return this._isInputTensorValue(argument.value[0]);
    }

    static _isTensorArgument(argument) {
        if (argument.type.includes("tensor") ) {
            return true;
        }
        if ( this._isPrimitiveArray(argument) && this._isTensorValue(argument.value[0]) ) {
            return true;
        }
        return false;
    }

    // -----------------------------
    // Edge Extraction
    // -----------------------------
    static _generateEdgeName(node) {
        return "start-node-x-edge";
    }

    static _getInputEdges(node) {
        const inputEdges = [];

        // If node is input-like (has direct 'value'), treat these as being driven by start-node aliases
        if ('value' in node && Array.isArray(node.value)) {
            for (const value of node.value) {
                if (!value) throw new Error('Invalid null argument in node.value.');
                inputEdges.push(`start-node-${value.name}`);
            }
        }

        const inputs = node.inputs;
        if (Array.isArray(inputs)) {
            for (const argument of inputs) {
                if ( !this._isPrimitiveArray(argument) )
                {
                    for (const value of argument.value) {
                        if (this._isInputTensorValue(value)) {
                            inputEdges.push(value.name);
                        }
                    }
                }
            }
        }

        if( inputEdges.length === 0 ) {
            inputEdges.push(this._generateEdgeName(node));
        }

        return inputEdges;
    }

    static _getOutputEdges(node) {
        const outputNames = new Set();

        // If node is input
        // TBC: Valid for Keras, TFLite, ONNX input
        if ('value' in node && Array.isArray(node.value)) {
            for (const value of node.value) {
                if (!value) {
                    throw new Error('Invalid null node value.');
                }
                outputNames.add(value.name);
            }
        } else {
            const addOutputsFromArgumentList = (outputsList) => {
                if (!Array.isArray(outputsList)) {
                    return;
                }
                for (const argument of outputsList) {
                    if (!Array.isArray(argument.value)) {
                        throw new Error('Invalid node output: argument.value must be an array.');
                    }
                    for (const value of argument.value) {
                        if (!value) {
                            throw new Error('Invalid null argument.');
                        }
                        outputNames.add(value.name);
                    }
                }
            };

            addOutputsFromArgumentList(node.outputs);

            // Incorporate chain outputs
            if (Array.isArray(node.chain) && node.chain.length > 0) {
                for (const stage of node.chain) {
                    if (stage && Array.isArray(stage.outputs)) {
                        addOutputsFromArgumentList(stage.outputs);
                    }
                }
            }
        }

        return Array.from(outputNames);
    }

    // -----------------------------
    // Model Inputs
    // -----------------------------

    static _getInputs(model) {
        const inputs = [];
        if (!model?.modules?.[0]?.inputs) {
            return inputs;
        }

        for (const input of model.modules[0].inputs) {
            if (!Array.isArray(input.value) || input.value.length === 0) {
                throw new Error('Invalid model input: value list missing.');
            }
            for (const v of input.value) {
                if (!v) {
                    throw new Error('Invalid null model input value.');
                }
            }
            inputs.push(input);
        }
        return inputs;
    }

    // -----------------------------
    // Edge Graph Construction
    // -----------------------------

    static _findEdges(model, startNode) {
        const edges = {};
        const inputs = this._getInputs(model);

        // Build edges for model inputs and their start-node aliases
        for (const input of inputs) {
            for (const val of input.value) {
                const startKey = `start-node-${val.name}`;

                if (!this._hasOwn(edges, startKey)) {
                    edges[startKey] = new diffAnalyser.Edge(startNode);
                }
                edges[startKey].addTo(input);

                if (!this._hasOwn(edges, val.name)) {
                    edges[val.name] = new diffAnalyser.Edge(input);
                }
            }
        }

        // Build edges across nodes
        const nodes = model?.modules?.[0]?.nodes ?? [];
                for (const node of nodes) {
            const outputEdges = this._getOutputEdges(node);
            for (const outputEdge of outputEdges) {
                edges[outputEdge] = new diffAnalyser.Edge(node);
            }

            const inputEdges = this._getInputEdges(node);
            
            let connectedToNode = false;
            let notConnectedEdges = []
            for (const inputEdge of inputEdges) {
                if (this._hasOwn(edges, inputEdge)) {
                    edges[inputEdge].addTo(node);
                    connectedToNode = true;
                } else {
                    notConnectedEdges.push(inputEdge)
                }
            }

            // TBC: Connect node without any connections to StartNode
            if (!connectedToNode && notConnectedEdges.length > 0) {
                for (const inputEdge of notConnectedEdges) {
                    if ( !( inputEdge in edges ) ) {
                        edges[inputEdge] = new diffAnalyser.Edge(startNode);
                        startNode.addEdge(inputEdge);
                    }

                    edges[inputEdge].addTo(node);
                }
            }
        }
        return edges;
    }

    static _getPyTorchAttributes(node) {
        let attrbs = new Map();
        const inputs = node.inputs;
        if (Array.isArray(inputs)) {
            for (const argument of inputs) {
                if (!this._isInputTensorArgument(argument)) {
                    attrbs.set(argument.name, argument);
                }
            }
        }               
        return attrbs;
    }

    static _getNodeAttributes(node) {
        let attrbs = new Map();
        for (const attr of node.attributes ?? []) {
            if (attr && attr.name) {
                attrbs.set(attr.name, attr);
            }
        }
        if( attrbs.size === 0 && node instanceof pytorch.Node) {
            attrbs = this._getPyTorchAttributes(node);
        }
        return attrbs;
    }

    static _isAttrbsSubset(node1, node2) {
        const attrbsSet1 = new Set([...this._getNodeAttributes(node1).keys()]);
        const attrbsSet2 = new Set([...this._getNodeAttributes(node2).keys()]);
        let matched = 0;
        for (const name of attrbsSet1) {
            if (attrbsSet2.has(name)) matched++;
        }

        // subset if all from the smaller set are present in the larger
        return matched === attrbsSet1.size || matched === attrbsSet2.size;
    }

    static _isNameMatch(node1, node2) {
        if (node1.name && node2.name) {
            return node1.name === node2.name;
        }
        else if (node1.identifier && node2.identifier) {
            return node1.identifier === node2.identifier;
        }
        return false;
    }

    static _getNodeTypeName(node) {
        // pytorch.Node
        if (typeof node.type === 'object' && 'name' in node.type) {
            return node.type.name;
        }
        return node.type;
    }

    static _isTypeMatch(node1, node2) {
        const typeName1 = this._getNodeTypeName(node1);
        const typeName2 = this._getNodeTypeName(node2);
        if (typeName1 && typeName2) {
            return typeName1 === typeName2;
        }
        return false;
    }
    
    static _isNodeMatch(node1, node2) {
        if (this._isTypeMatch(node1, node2)) {
            return this._isNameMatch(node1, node2) || this._isAttrbsSubset(node1, node2);
        } 
        else {
            if (this._isNameMatch(node1, node2)) {
                return this._isAttrbsSubset(node1, node2);
            }
        }
        return false;
    }

    static _getParentNodes(node, edges) {
        const parentNodes = [];
        const inputEdges = this._getInputEdges(node);
        for (const inputEdge of inputEdges) {
            if (this._hasOwn(edges, inputEdge)) {
                const edge = edges[inputEdge];
                parentNodes.push(edge.from);
            }
        }
        return parentNodes;
    }

    static _getChildNodes(node, edges) {
        const childNodes = [];
        const outputEdges = this._getOutputEdges(node);
        for (const outputEdge of outputEdges) {
            if (this._hasOwn(edges, outputEdge)) {
                const edge = edges[outputEdge];
                if (edge.to === null) continue;
                for (const toNode of edge.to) {
                    childNodes.push(toNode);
                }
            }
        }
        return childNodes;
    }

    static _findMatchChildNode(targetMatchNode1, parentNode2, edgesModel2, nameIndex2, usedNodes2, visited = new Set()) 
    {
        if (!parentNode2) {
            return [null, null];
        }
        if (visited.has(parentNode2)) {
            return [null, null];
        }
        visited.add(parentNode2);

        const childNodes = this._getChildNodes(parentNode2, edgesModel2);
        if (childNodes && childNodes.some(Boolean)) {
            // Prefer direct children first, skip already-used candidates
            for (const child of childNodes) {
                if (!child) { 
                    continue;
                }
                if (usedNodes2.has(child)) {
                    continue;
                }
                if (this._isNodeMatch(targetMatchNode1, child)) {
                    const nodeDifferences = diffAnalyser.NodeDiffAnalyzer.compare(targetMatchNode1, child);
                    return [child, nodeDifferences];
                }
            }

            // DFS deeper, still respecting usedNodes2 and cycle safety
            for (const child of childNodes) {
                if (!child) continue;
                const result = this._findMatchChildNode(targetMatchNode1, child, edgesModel2, nameIndex2, usedNodes2, visited);
                if (result && result[0]) {
                    return result;
                }
            }
        }

        return [null, null];
    }


    static _isMatchedNode(node, nodeToNode) {
        return nodeToNode.has(node);
    }

    static _findMatchedParentNode(node, edges, nodeToNode, visited = new Set()) {
        if (visited.has(node)) {
            return null;
        }
        visited.add(node);

        const parentNodes = this._getParentNodes(node, edges);

        for (const parent of parentNodes) {
            if (this._isMatchedNode(parent, nodeToNode)) {
                return parent;
            }
        }

        for (const parent of parentNodes) {
            const result = this._findMatchedParentNode(parent, edges, nodeToNode, visited);
            if (result) {
                return result;
            }
        }

        return null;
    }

    static _isInputMatch(input1, input2) {
        return input1.name === input2.name;
    }

    static _getInputToInputMapping(model1, model2) {
        const inputToInput = new Map();

        const model1Inputs = this._getInputs(model1);
        const model2Inputs = this._getInputs(model2);

        for (const input1 of model1Inputs) {
            for (const input2 of model2Inputs) {
                if (this._isInputMatch(input1, input2)) {
                    inputToInput.set(input1, input2);
                    break;
                }
            }
        }
        return inputToInput;
    }

    // -----------------------------
    // Main Compare
    // -----------------------------

    static compare(model1, model2) {
        if (model1 === null || model2 === null) {
            return new diffAnalyser.ModelDifferences([]);
        }

        const model1Inputs = this._getInputs(model1);
        const model2Inputs = this._getInputs(model2);

        const model1StartNode = new diffAnalyser.StartNode(model1Inputs);
        const model2StartNode = new diffAnalyser.StartNode(model2Inputs);

        const nodeToNode = this._getInputToInputMapping(model1, model2);
        nodeToNode.set(model1StartNode, model2StartNode);

        const model1Edges = this._findEdges(model1, model1StartNode);
        const model2Edges = this._findEdges(model2, model2StartNode);
        
        // Track nodes in model2 that have already been matched to avoid duplicate matches
        const usedNodes2 = new Set();
        usedNodes2.add(model2StartNode);

        // Build a name -> [nodes] index for model2 to accelerate matching
        const nodes2 = model2?.modules?.[0]?.nodes ?? [];
        const nameIndex2 = new Map();
        for (const n2 of nodes2) {
            const k = n2 && typeof n2.name === 'string' ? n2.name : '__UNNAMED__';
            if (!nameIndex2.has(k)) nameIndex2.set(k, []);
            nameIndex2.get(k).push(n2);
        }

        let index = 0;
        const modelDiffEntries = [];
        const nodes1 = model1?.modules?.[0]?.nodes ?? [];

        for (const node1 of nodes1) {
            const parentNode1 = this._findMatchedParentNode(node1, model1Edges, nodeToNode, new Set());
            const parentNode2 = parentNode1 ? nodeToNode.get(parentNode1) : model2StartNode;

            const [node2, nodeDifferences] = this._findMatchChildNode(
                node1,
                parentNode2,
                model2Edges,
                nameIndex2,
                usedNodes2,
                new Set()
            );

            if (node2 !== null) {
                const id = `node-${index}`;
                const modelDiffEntry = new diffAnalyser.ModelNodesDiffAnalyzerEntry(id, node1, node2, nodeDifferences);
                modelDiffEntries.push(modelDiffEntry);
                nodeToNode.set(node1, node2);
                usedNodes2.add(node2)
                index++;
            }
        }

        return new diffAnalyser.ModelDifferences(modelDiffEntries);
    }
};

diffAnalyser.SoftModelNodesDiffAnalyzer = class {

    // ----------------------------- Constants -----------------------------
    static FEATURE_DIM = 19;
    static MESSAGE_PASS_ROUNDS = 4;
    static SOFT_MATCH_THRESHOLD = 0.35;
    static EXPANSION_THRESHOLD = 0.25;
    static SELF_WEIGHT = 0.6;
    // Upper bound for type-ordinal normalization (vec[18]).
    // Conv#0 → 0/MAX, Conv#1 → 1/MAX, … same index = same value across both models.
    static MAX_TYPE_COUNT = 100;
    // Multiplier for vec[0] (node typeIdx). Values > 1 increase its weight
    // in cosine similarity relative to all other structural features.
    static TYPE_IDX_SCALE = 3.0;

    // ----------------------------- Type Vocabulary -----------------------------

    static _buildVocabulary(nodes1, nodes2) {
        const typeToIdx = new Map();
        for (const node of [...nodes1, ...nodes2]) {
            const t = diffAnalyser.ModelNodesDiffAnalyzer._getNodeTypeName(node) ?? '__unknown__';
            if (!typeToIdx.has(t)) {
                typeToIdx.set(t, typeToIdx.size);
            }
        }
        return { typeToIdx, size: typeToIdx.size || 1 };
    }

    static _buildCompatibilityGroups(vocab) {
        const groups = new Int32Array(vocab.size);

        const rules = [
            [/conv/i,                              0],  // conv2d, convolution, conv_transpose, depthwise_conv
            [/batchnorm|batch_norm|^bn$/i,         1],  // BatchNorm2d, batch_normalization, FusedBatchNorm
            [/relu|activation|sigmoid|tanh|^elu|leaky|hardswish|gelu|silu|mish/i, 2],
            [/pool/i,                              3],  // MaxPool, AvgPool, GlobalAveragePool
            [/linear|dense|gemm|matmul|^fc$|fully_connected|inner_product/i, 4],
            [/^add$|^mul$|^div$|concat|merge|elementwise|add_n|multiply/i,   5],
            [/dropout/i,                                                     6],  // Dropout, AlphaDropout, DropPath
        ];

        // Known groups occupy 0–6. Unmatched types each get a unique group
        // (100 + typeIdx) so they are never compatible with each other.
        const UNKNOWN_BASE = 100;
        for (const [typeName, idx] of vocab.typeToIdx) {
            let group = UNKNOWN_BASE + idx;
            for (const [re, g] of rules) {
                if (re.test(typeName)) { group = g; break; }
            }
            groups[idx] = group;
        }
        return groups;
    }

    // ----------------------------- Graph Topology -----------------------------

    static _buildTopology(nodes, edges) {
        const MNA = diffAnalyser.ModelNodesDiffAnalyzer;
        const parentMap = new Map();
        const childMap  = new Map();
        const inDegree  = new Map();
        const outDegree = new Map();

        for (const node of nodes) {
            const parents  = MNA._getParentNodes(node, edges).filter(p => !(p instanceof diffAnalyser.StartNode));
            const children = MNA._getChildNodes(node, edges).filter(c => !(c instanceof diffAnalyser.StartNode));
            parentMap.set(node, parents);
            childMap.set(node,  children);
            inDegree.set(node,  parents.length);
            outDegree.set(node, children.length);
        }

        // Kahn's BFS topological ranking
        const topoRank = new Map();
        const pending  = new Map(nodes.map(n => [n, parentMap.get(n).filter(p => nodes.includes(p)).length]));
        let queue = nodes.filter(n => pending.get(n) === 0);
        let rank  = 0;
        while (queue.length > 0) {
            const next = [];
            for (const node of queue) {
                topoRank.set(node, rank);
                for (const child of (childMap.get(node) ?? [])) {
                    if (!pending.has(child)) continue;
                    const cnt = pending.get(child) - 1;
                    pending.set(child, cnt);
                    if (cnt === 0) next.push(child);
                }
            }
            rank++;
            queue = next;
        }
        // Nodes unreachable (cycles) get rank 0
        for (const node of nodes) {
            if (!topoRank.has(node)) topoRank.set(node, 0);
        }

        return { parentMap, childMap, inDegree, outDegree, topoRank };
    }

    // ----------------------------- Feature Vectors -----------------------------

    // Helper: read dimensions from node.inputs[inputIdx].value[0].type.shape.dimensions
    // Returns null if unavailable or dimensions has wrong rank.
    static _weightDims(node, inputIdx) {
        const arg = (node.inputs ?? [])[inputIdx];
        const dims = arg?.value?.[0]?.type?.shape?.dimensions;
        return Array.isArray(dims) && dims.length >= 2 ? dims : null;
    }

    // ONNX: attribute values are BigInt or BigInt[] (INT/INTS proto fields).
    // kernel_shape and strides come from op attributes.
    // Weight tensor (inputs[1]) layout: [out_ch, in_ch, kH, kW]
    static _extractAttrFeaturesOnnx(node) {
        const toNum = (v) => (typeof v === 'bigint') ? Number(v) : (typeof v === 'number' ? v : 0);
        const attrs = new Map();
        for (const a of (node.attributes ?? [])) { if (a?.name) attrs.set(a.name, a); }
        const arr = (key) => {
            const v = attrs.get(key)?.value;
            if (Array.isArray(v)) return v.map(toNum);
            if (v != null) return [toNum(v)];
            return null;
        };
        const scalar = (key) => { const a = arr(key); return a ? a[0] : null; };
        const ks = arr('kernel_shape');
        const st = arr('strides');

        // kernel_shape is explicit for Conv; for Pool it may be absent — fall back to weight tensor
        let ks0 = ks ? ks[0] : null;
        let ks1 = ks ? (ks[1] ?? ks[0]) : null;
        let outCh = null;
        const dims = this._weightDims(node, 1); // [out_ch, in_ch, kH, kW]
        if (dims && dims.length >= 4) {
            if (ks0 === null) ks0 = toNum(dims[2]);
            if (ks1 === null) ks1 = toNum(dims[3]);
            outCh = toNum(dims[0]);
        }

        return {
            ks0,
            ks1,
            stride:    st ? st[0] : null,
            outCh,
            groups:    scalar('group'),
            attrCount: attrs.size,
        };
    }

    // PyTorch: numeric params live in node.inputs (not node.attributes).
    // Weight tensor (inputs[1]) layout: [out_ch, in_ch/groups, kH, kW]
    static _extractAttrFeaturesPytorch(node) {
        const toNum = (v) => (typeof v === 'bigint') ? Number(v) : (typeof v === 'number' ? v : (Number(v) || 0));
        const attrs = diffAnalyser.ModelNodesDiffAnalyzer._getPyTorchAttributes(node);
        const first = (keys) => {
            for (const k of keys) {
                const v = attrs.get(k)?.value;
                if (v == null) continue;
                if (Array.isArray(v) && v.length > 0) return toNum(v[0]);
                return toNum(v);
            }
            return null;
        };
        const arr = (keys) => {
            for (const k of keys) {
                const v = attrs.get(k)?.value;
                if (v == null) continue;
                if (Array.isArray(v)) return v.map(toNum);
                return [toNum(v)];
            }
            return null;
        };
        const ks = arr(['kernel_size', 'kernel_sizes']);

        let ks0   = ks ? ks[0] : null;
        let ks1   = ks ? (ks[1] ?? ks[0]) : null;
        let outCh = first(['out_channels', 'out_features', 'num_features']);

        // Fall back to weight tensor shape for ops without explicit attribute params
        const dims = this._weightDims(node, 1); // [out_ch, in_ch/groups, kH, kW]
        if (dims && dims.length >= 4) {
            if (ks0   === null) ks0   = toNum(dims[2]);
            if (ks1   === null) ks1   = toNum(dims[3]);
            if (outCh === null) outCh = toNum(dims[0]);
        }

        return {
            ks0,
            ks1,
            stride: first(['stride', 'strides']),
            outCh,
            groups: first(['groups', 'group']),
            attrCount: attrs.size,
        };
    }

    // TFLite: builtin_options are decoded as flat scalars via tflite-schema.js.
    // Attribute names by op:
    //   Conv2D / TransposeConv: stride_w, stride_h, dilation_w_factor, dilation_h_factor
    //   DepthwiseConv2D:        stride_w, stride_h, dilation_w_factor, dilation_h_factor, depth_multiplier
    //   Pool2D (Avg/Max):       stride_w, stride_h, filter_width, filter_height
    // Filter size for Conv ops is stored in the weight tensor shape (inputs[1]):
    //   Conv2D weight layout:         [out_channels, kH, kW, in_channels]
    //   DepthwiseConv2D weight layout: [1, kH, kW, depth_multiplier]
    static _extractAttrFeaturesTflite(node) {
        const toNum = (v) => (typeof v === 'bigint') ? Number(v) : (typeof v === 'number' ? v : (Number(v) || 0));
        const attrs = new Map();
        for (const a of (node.attributes ?? [])) { if (a?.name) attrs.set(a.name, a); }
        const scalar = (keys) => {
            for (const k of keys) {
                const v = attrs.get(k)?.value;
                if (v != null && v !== 0) return toNum(v);
            }
            return null;
        };

        // Pool ops expose kernel size directly as attributes
        let ks0 = scalar(['filter_height']);
        let ks1 = scalar(['filter_width']);
        let outCh = scalar(['depth_multiplier']);

        // Conv ops: read kernel size and output channels from weight tensor shape (inputs[1])
        if (ks0 === null || ks1 === null || outCh === null) {
            const weightArg = (node.inputs ?? [])[1];
            const dims = weightArg?.value?.[0]?.type?.shape?.dimensions;
            if (Array.isArray(dims) && dims.length === 4) {
                // [out_channels, kH, kW, in_channels] for Conv2D
                // [1, kH, kW, depth_multiplier] for DepthwiseConv2D
                if (ks0 === null) ks0 = toNum(dims[1]);
                if (ks1 === null) ks1 = toNum(dims[2]);
                if (outCh === null) outCh = toNum(dims[0]);
            }
        }

        return {
            ks0,
            ks1,
            stride: scalar(['stride_w', 'stride_h']),
            outCh,
            groups: null,
            attrCount: attrs.size,
        };
    }

    // TF/PB: strides and ksize stored as NHWC lists [1, h, w, 1].
    // Filter tensor (inputs[1]) layout: [kH, kW, in_ch, out_ch]
    static _extractAttrFeaturesTf(node) {
        const toNum = (v) => {
            if (typeof v === 'bigint') return Number(v);
            if (typeof v === 'number') return v;
            if (v === '?') return 0;
            return Number(v) || 0;
        };
        const attrs = new Map();
        for (const a of (node.attributes ?? [])) { if (a?.name) attrs.set(a.name, a); }
        const nhwcIdx = (key, idx) => {
            const v = attrs.get(key)?.value;
            if (Array.isArray(v) && v.length > idx) return toNum(v[idx]);
            if (v?.dimensions && v.dimensions.length > idx) return toNum(v.dimensions[idx]);
            return null;
        };

        // strides/ksize format is [1, h, w, 1] (NHWC); index 1 = height, 2 = width
        let ks0   = nhwcIdx('ksize', 1);
        let ks1   = nhwcIdx('ksize', 2);
        let outCh = null;

        // Filter weight tensor (inputs[1]): [kH, kW, in_ch, out_ch]
        const dims = this._weightDims(node, 1);
        if (dims && dims.length >= 4) {
            if (ks0   === null) ks0   = toNum(dims[0]);
            if (ks1   === null) ks1   = toNum(dims[1]);
            outCh = toNum(dims[3]);
        }

        return {
            ks0,
            ks1,
            stride: nhwcIdx('strides', 1),
            outCh,
            groups: null,
            attrCount: attrs.size,
        };
    }

    // Keras: kernel_size, strides, filters come from the layer config (JSON attributes).
    // Kernel weight tensor (inputs[1]) layout: [kH, kW, in_ch, out_ch]
    // Config attributes are preferred; weight tensor is a fallback for custom layers.
    static _extractAttrFeaturesKeras(node) {
        const toNum = (v) => (typeof v === 'bigint') ? Number(v) : (typeof v === 'number' ? v : (Number(v) || 0));
        const attrs = new Map();
        for (const a of (node.attributes ?? [])) { if (a?.name) attrs.set(a.name, a); }
        const first = (keys) => {
            for (const k of keys) {
                const v = attrs.get(k)?.value;
                if (v == null) continue;
                if (Array.isArray(v) && v.length > 0) return toNum(v[0]);
                if (typeof v === 'number' || typeof v === 'bigint') return toNum(v);
            }
            return null;
        };
        const arr = (keys) => {
            for (const k of keys) {
                const v = attrs.get(k)?.value;
                if (v == null) continue;
                if (Array.isArray(v)) return v.map(toNum);
                if (typeof v === 'number') return [toNum(v)];
            }
            return null;
        };
        const ks = arr(['kernel_size', 'kernel_sizes']);
        const st = arr(['strides', 'stride']);

        let ks0   = ks ? ks[0] : null;
        let ks1   = ks ? (ks[1] ?? ks[0]) : null;
        let outCh = first(['filters', 'units', 'num_filters', 'out_channels']);

        // Fallback: kernel weight tensor (inputs[1]): [kH, kW, in_ch, out_ch]
        const dims = this._weightDims(node, 1);
        if (dims && dims.length >= 4) {
            if (ks0   === null) ks0   = toNum(dims[0]);
            if (ks1   === null) ks1   = toNum(dims[1]);
            if (outCh === null) outCh = toNum(dims[3]);
        }

        return {
            ks0,
            ks1,
            stride: st ? st[0] : null,
            outCh,
            groups: first(['groups', 'group']),
            attrCount: attrs.size,
        };
    }

    // Default fallback for all other formats.
    // Tries common attribute names first, then weight tensor (inputs[1]) as fallback.
    // Assumes the most common Conv weight layout [out_ch, in_ch, kH, kW] (ONNX-style).
    static _extractAttrFeaturesDefault(node) {
        const toNum = (v) => (typeof v === 'bigint') ? Number(v) : (typeof v === 'number' ? v : (Number(v) || 0));
        const attrs = diffAnalyser.ModelNodesDiffAnalyzer._getNodeAttributes(node);
        const first = (keys) => {
            for (const k of keys) {
                const v = attrs.get(k)?.value;
                if (v == null) continue;
                if (Array.isArray(v) && v.length > 0) return toNum(v[0]);
                if (v !== null && v !== undefined) return toNum(v);
            }
            return null;
        };

        let ks0   = first(['kernel_shape', 'kernel_size', 'filter_height']);
        let ks1   = first(['kernel_shape', 'kernel_size', 'filter_width']);
        let outCh = first(['out_channels', 'num_output', 'num_filters', 'filters', 'units']);

        // Fallback to weight tensor shape (inputs[1]), assuming [out_ch, in_ch, kH, kW]
        const dims = this._weightDims(node, 1);
        if (dims && dims.length >= 4) {
            if (ks0   === null) ks0   = toNum(dims[2]);
            if (ks1   === null) ks1   = toNum(dims[3]);
            if (outCh === null) outCh = toNum(dims[0]);
        }

        return {
            ks0,
            ks1,
            stride: first(['strides', 'stride', 'stride_w', 'stride_h']),
            outCh,
            groups: first(['group', 'groups']),
            attrCount: attrs.size,
        };
    }

    static _extractAttrFeatures(node) {
        if (node instanceof onnx.Node)    return this._extractAttrFeaturesOnnx(node);
        if (node instanceof pytorch.Node) return this._extractAttrFeaturesPytorch(node);
        if (node instanceof tflite.Node)  return this._extractAttrFeaturesTflite(node);
        if (node instanceof tf.Node)      return this._extractAttrFeaturesTf(node);
        if (node instanceof keras.Node)   return this._extractAttrFeaturesKeras(node);
        return this._extractAttrFeaturesDefault(node);
    }

    static _buildFeatureVectors(nodes, topo, vocab) {
        const DIM    = this.FEATURE_DIM;
        const maxDeg = Math.max(1, ...nodes.map(n => Math.max(topo.inDegree.get(n) ?? 0, topo.outDegree.get(n) ?? 0)));
        const maxRank = Math.max(1, ...[...topo.topoRank.values()]);
        const fmap = new Map();

        const MNA = diffAnalyser.ModelNodesDiffAnalyzer;

        // Absolute ordinal index among nodes of the same type, sorted by topoRank.
        // Normalized by MAX_TYPE_COUNT (fixed constant, not per-model count) so that
        // Conv#2 in model1 and Conv#2 in model2 produce identical vec[18] values.
        const typeRelRank = new Map();
        const byType = new Map();
        for (const node of nodes) {
            const tn = MNA._getNodeTypeName(node) ?? '__unknown__';
            if (!byType.has(tn)) byType.set(tn, []);
            byType.get(tn).push(node);
        }
        for (const group of byType.values()) {
            group.sort((a, b) => (topo.topoRank.get(a) ?? 0) - (topo.topoRank.get(b) ?? 0));
            group.forEach((node, i) => typeRelRank.set(node, i / this.MAX_TYPE_COUNT));
        }

        const meanTypeIdx = (neighbors) => {
            let sum = 0, cnt = 0;
            for (const nbr of neighbors) {
                if (!nbr || nbr.type == null) continue;
                const tn = MNA._getNodeTypeName(nbr) ?? '__unknown__';
                sum += vocab.typeToIdx.get(tn) ?? 0;
                cnt++;
            }
            return cnt > 0 ? sum / cnt : 0;
        };

        for (const node of nodes) {
            const vec = new Float32Array(DIM);
            const typeName = MNA._getNodeTypeName(node) ?? '__unknown__';
            const typeIdx  = vocab.typeToIdx.get(typeName) ?? 0;
            const { ks0, ks1, stride, outCh, groups, attrCount } = this._extractAttrFeatures(node);

            vec[0]  = typeIdx / vocab.size * this.TYPE_IDX_SCALE;
            vec[1]  = (topo.inDegree.get(node)  ?? 0) / maxDeg;
            vec[2]  = (topo.outDegree.get(node) ?? 0) / maxDeg;
            vec[3]  = (topo.topoRank.get(node)  ?? 0) / maxRank;
            vec[4]  = ks0 !== null ? 1 : 0;
            vec[5]  = ks0 !== null ? Math.min(ks0, 32) / 32 : 0;
            vec[6]  = ks1 !== null ? Math.min(ks1, 32) / 32 : 0;
            vec[7]  = stride !== null ? 1 : 0;
            vec[8]  = stride !== null ? Math.min(stride, 8) / 8 : 0;
            vec[9]  = outCh !== null ? 1 : 0;
            vec[10] = outCh !== null ? Math.min(outCh, 4096) / 4096 : 0;
            vec[11] = groups !== null ? 1 : 0;
            vec[12] = groups !== null ? Math.min(groups, 256) / 256 : 0;
            vec[13] = Math.min(attrCount, 20) / 20;
            vec[14] = Math.min((node.inputs  ?? []).length, 8) / 8;
            vec[15] = Math.min((node.outputs ?? []).length, 8) / 8;
            vec[16] = meanTypeIdx(topo.parentMap.get(node) ?? []) / vocab.size;
            vec[17] = meanTypeIdx(topo.childMap.get(node)  ?? []) / vocab.size;
            vec[18] = typeRelRank.get(node) ?? 0;

            fmap.set(node, vec);
        }
        return fmap;
    }

    // ----------------------------- Message Passing -----------------------------

    // Weights for directed message passing: self + upstream (parents) + downstream (children).
    // W_PAR != W_CHI encodes edge direction — relu→conv→bn differs from conv→bn→relu.
    static W_PAR   = 0.3; // 0.3 — upstream context
    static W_CHI   = 0.2; // 0.2 — downstream context

    static _messagePass(nodes, fmap, topo, rounds) {
        const DIM   = this.FEATURE_DIM;
        const WSELF = this.SELF_WEIGHT;
        const WPAR  = this.W_PAR;
        const WCHI  = this.W_CHI;

        const meanVec = (neighbors) => {
            const acc = new Float32Array(DIM);
            let cnt = 0;
            for (const nbr of neighbors) {
                const nv = fmap.get(nbr);
                if (!nv) continue;
                for (let i = 0; i < DIM; i++) acc[i] += nv[i];
                cnt++;
            }
            if (cnt > 1) for (let i = 0; i < DIM; i++) acc[i] /= cnt;
            return { vec: acc, any: cnt > 0 };
        };

        for (let r = 0; r < rounds; r++) {
            const newMap = new Map();
            for (const node of nodes) {
                const selfVec = fmap.get(node);
                const { vec: parVec,  any: hasPar  } = meanVec(topo.parentMap.get(node) ?? []);
                const { vec: chiVec,  any: hasChi  } = meanVec(topo.childMap.get(node)  ?? []);

                const newVec = new Float32Array(DIM);
                for (let i = 0; i < DIM; i++) {
                    newVec[i] = WSELF * selfVec[i]
                              + (hasPar ? WPAR * parVec[i] : 0)
                              + (hasChi ? WCHI * chiVec[i] : 0);
                    // Renormalize when a direction is absent so weights still sum to 1
                    if (!hasPar && !hasChi) {
                        newVec[i] = selfVec[i];
                    } else if (!hasPar) {
                        newVec[i] = WSELF / (WSELF + WCHI) * selfVec[i]
                                  + WCHI  / (WSELF + WCHI) * chiVec[i];
                    } else if (!hasChi) {
                        newVec[i] = WSELF / (WSELF + WPAR) * selfVec[i]
                                  + WPAR  / (WSELF + WPAR) * parVec[i];
                    }
                }
                newMap.set(node, newVec);
            }
            fmap = newMap;
        }
        return fmap;
    }

    // ----------------------------- Cosine Similarity + Type Mask -----------------------------

    static _l2norm(vec) {
        let s = 0;
        for (let i = 0; i < vec.length; i++) s += vec[i] * vec[i];
        return Math.sqrt(s);
    }

    static _buildSimilarityMatrix(nodes1, nodes2, fmap1, fmap2, compatGroups, vocab) {
        const n1  = nodes1.length;
        const n2  = nodes2.length;
        const DIM = this.FEATURE_DIM;
        const sim = new Float32Array(n1 * n2);

        // Precompute norms and group for nodes2
        const norms2  = new Float32Array(n2);
        const groups2 = new Int32Array(n2);
        for (let j = 0; j < n2; j++) {
            norms2[j]  = this._l2norm(fmap2.get(nodes2[j]));
            const tn   = diffAnalyser.ModelNodesDiffAnalyzer._getNodeTypeName(nodes2[j]) ?? '__unknown__';
            groups2[j] = compatGroups[vocab.typeToIdx.get(tn) ?? 0] ?? 99;
        }

        for (let i = 0; i < n1; i++) {
            const vec1   = fmap1.get(nodes1[i]);
            const norm1  = this._l2norm(vec1);
            const tn1    = diffAnalyser.ModelNodesDiffAnalyzer._getNodeTypeName(nodes1[i]) ?? '__unknown__';
            const group1 = compatGroups[vocab.typeToIdx.get(tn1) ?? 0] ?? 99;

            for (let j = 0; j < n2; j++) {
                if (group1 !== groups2[j]) {
                    // type-incompatible: hard zero
                    sim[i * n2 + j] = 0;
                    continue;
                }
                if (norm1 < 1e-9 || norms2[j] < 1e-9) {
                    sim[i * n2 + j] = 0;
                    continue;
                }
                const vec2 = fmap2.get(nodes2[j]);
                let dot = 0;
                for (let k = 0; k < DIM; k++) dot += vec1[k] * vec2[k];
                sim[i * n2 + j] = dot / (norm1 * norms2[j]);
            }
        }
        return sim;
    }

    // ----------------------------- Greedy Strong-First Matching -----------------------------
    //
    // Divide-and-conquer on subgraph pools:
    //   matchSubgraphs(pool1, pool2):
    //     1. Find the best pair (seed) in pool1 × pool2 — assign it
    //     2. Partition each pool around the seed:
    //          ancestors of seed  → recurse matchSubgraphs(anc1,  anc2)
    //          descendants of seed → recurse matchSubgraphs(desc1, desc2)
    //          remainder (disconnected from seed within the pool) → recurse matchSubgraphs(rem1, rem2)
    //     3. Base case: no pair above threshold found → return

    static _greedyMatch(nodes1, nodes2, sim, threshold, topo1, topo2) {
        const n2 = nodes2.length;

        const idx1Map = new Map(nodes1.map((n, i) => [n, i]));
        const idx2Map = new Map(nodes2.map((n, i) => [n, i]));

        const matched1    = new Set();
        const matched2    = new Set();
        const assignments = [];

        // Transitive ancestors of nodes[nodeIdx], restricted to the given pool
        const ancestorPool = (nodeIdx, nodes, topo, idxMap, pool) => {
            const result = new Set();
            const stack  = [...(topo.parentMap.get(nodes[nodeIdx]) ?? [])];
            while (stack.length > 0) {
                const node = stack.pop();
                const ni   = idxMap.get(node);
                if (ni === undefined || result.has(ni)) continue;
                result.add(ni);
                for (const p of (topo.parentMap.get(node) ?? [])) stack.push(p);
            }
            // Intersect with the current pool so we never escape our subgraph boundary
            for (const ni of result) { if (!pool.has(ni)) result.delete(ni); }
            return result;
        };

        // Transitive descendants of nodes[nodeIdx], restricted to the given pool
        const descendantPool = (nodeIdx, nodes, topo, idxMap, pool) => {
            const result = new Set();
            const stack  = [...(topo.childMap.get(nodes[nodeIdx]) ?? [])];
            while (stack.length > 0) {
                const node = stack.pop();
                const ni   = idxMap.get(node);
                if (ni === undefined || result.has(ni)) continue;
                result.add(ni);
                for (const c of (topo.childMap.get(node) ?? [])) stack.push(c);
            }
            for (const ni of result) { if (!pool.has(ni)) result.delete(ni); }
            return result;
        };

        // Best unmatched pair in pool1 × pool2 above threshold; null if none found
        const bestInPools = (pool1, pool2) => {
            let bestI = -1, bestJ = -1, bestScore = threshold;
            for (const ni of pool1) {
                if (matched1.has(ni)) continue;
                for (const nj of pool2) {
                    if (matched2.has(nj)) continue;
                    const s = sim[ni * n2 + nj];
                    if (s > bestScore) { bestScore = s; bestI = ni; bestJ = nj; }
                }
            }
            return bestI !== -1 ? { i: bestI, j: bestJ } : null;
        };

        const assign = (i, j) => {
            matched1.add(i);
            matched2.add(j);
            assignments.push({ idx1: i, idx2: j, score: sim[i * n2 + j] });
        };

        // TODO: GW scoring boost could be inserted here for pairs with low sim score

        const matchSubgraphs = (pool1, pool2) => {
            // Base case: find best seed in this subgraph
            const seed = bestInPools(pool1, pool2);
            if (!seed) return;
            assign(seed.i, seed.j);

            // Ancestor subgraphs — nodes that feed into the seed
            const anc1 = ancestorPool(seed.i, nodes1, topo1, idx1Map, pool1);
            const anc2 = ancestorPool(seed.j, nodes2, topo2, idx2Map, pool2);
            if (anc1.size > 0 && anc2.size > 0) matchSubgraphs(anc1, anc2);

            // Descendant subgraphs — nodes that the seed feeds into
            const desc1 = descendantPool(seed.i, nodes1, topo1, idx1Map, pool1);
            const desc2 = descendantPool(seed.j, nodes2, topo2, idx2Map, pool2);
            if (desc1.size > 0 && desc2.size > 0) matchSubgraphs(desc1, desc2);

            // Remainder — nodes disconnected from the seed within this pool
            // (handles parallel branches and isolated subgraphs)
            const rem1 = new Set([...pool1].filter(x => x !== seed.i && !anc1.has(x) && !desc1.has(x)));
            const rem2 = new Set([...pool2].filter(x => x !== seed.j && !anc2.has(x) && !desc2.has(x)));
            if (rem1.size > 0 && rem2.size > 0) matchSubgraphs(rem1, rem2);
        };

        matchSubgraphs(
            new Set(nodes1.map((_, i) => i)),
            new Set(nodes2.map((_, i) => i))
        );

        return { assignments, matched1, matched2 };
    }

    // ----------------------------- Local Graph Expansion -----------------------------

    static _expandMatches(assignments, matched1, matched2, nodes1, nodes2, topo1, topo2, sim, n2) {
        const THRESH = this.EXPANSION_THRESHOLD;

        const idx1Map   = new Map(nodes1.map((n, i) => [n, i]));
        const idx2Map   = new Map(nodes2.map((n, i) => [n, i]));
        const assignMap = new Map(assignments.map(a => [a.idx1, a.idx2]));

        const bestCand = (i, anchors1, getCands2) => {
            let bestScore = THRESH, bestJ = -1;
            for (const nbr1 of anchors1) {
                const nbrIdx1 = idx1Map.get(nbr1);
                if (nbrIdx1 === undefined || !assignMap.has(nbrIdx1)) continue;
                for (const cand2 of getCands2(nodes2[assignMap.get(nbrIdx1)])) {
                    const j = idx2Map.get(cand2);
                    if (j === undefined || matched2.has(j)) continue;
                    const s = sim[i * n2 + j];
                    if (s > bestScore) { bestScore = s; bestJ = j; }
                }
            }
            return bestJ;
        };

        let changed = true;
        while (changed) {
            changed = false;
            for (let i = 0; i < nodes1.length; i++) {
                if (matched1.has(i)) continue;
                const node1 = nodes1[i];

                // node1 comes AFTER its parents → its match must come after the matched parent
                let bestJ = bestCand(i,
                    topo1.parentMap.get(node1) ?? [],
                    n2nbr => topo2.childMap.get(n2nbr) ?? []
                );

                // node1 comes BEFORE its children → its match must come before the matched child
                if (bestJ === -1) {
                    bestJ = bestCand(i,
                        topo1.childMap.get(node1) ?? [],
                        n2nbr => topo2.parentMap.get(n2nbr) ?? []
                    );
                }

                if (bestJ !== -1) {
                    assignments.push({ idx1: i, idx2: bestJ, score: sim[i * n2 + bestJ] });
                    matched1.add(i);
                    matched2.add(bestJ);
                    assignMap.set(i, bestJ);
                    changed = true;
                }
            }
        }
    }

    // ----------------------------- Build Output -----------------------------

    static _buildModelDifferences(assignments, nodes1, nodes2) {
        const entries = [];
        let index = 0;
        for (const { idx1, idx2 } of assignments) {
            const node1          = nodes1[idx1];
            const node2          = nodes2[idx2];
            const nodeDifferences = diffAnalyser.NodeDiffAnalyzer.compare(node1, node2);
            const id             = `soft-node-${index++}`;
            entries.push(new diffAnalyser.ModelNodesDiffAnalyzerEntry(id, node1, node2, nodeDifferences));
        }
        return new diffAnalyser.ModelDifferences(entries);
    }

    // ----------------------------- Main Entry Point -----------------------------

    static compare(model1, model2) {
        if (model1 === null || model2 === null) {
            return new diffAnalyser.ModelDifferences([]);
        }

        const nodes1 = model1?.modules?.[0]?.nodes ?? [];
        const nodes2 = model2?.modules?.[0]?.nodes ?? [];

        if (nodes1.length === 0 && nodes2.length === 0) {
            return new diffAnalyser.ModelDifferences([]);
        }

        // 1. Shared type vocabulary
        const vocab        = this._buildVocabulary(nodes1, nodes2);
        const compatGroups = this._buildCompatibilityGroups(vocab);

        // 2. Edge graphs (reuse existing infrastructure)
        const model1Inputs    = diffAnalyser.ModelNodesDiffAnalyzer._getInputs(model1);
        const model2Inputs    = diffAnalyser.ModelNodesDiffAnalyzer._getInputs(model2);
        const model1StartNode = new diffAnalyser.StartNode(model1Inputs);
        const model2StartNode = new diffAnalyser.StartNode(model2Inputs);
        const edges1 = diffAnalyser.ModelNodesDiffAnalyzer._findEdges(model1, model1StartNode);
        const edges2 = diffAnalyser.ModelNodesDiffAnalyzer._findEdges(model2, model2StartNode);

        // 3. Topology
        const topo1 = this._buildTopology(nodes1, edges1);
        const topo2 = this._buildTopology(nodes2, edges2);

        // 4. Feature vectors + message passing
        let fmap1 = this._buildFeatureVectors(nodes1, topo1, vocab);
        let fmap2 = this._buildFeatureVectors(nodes2, topo2, vocab);
        fmap1 = this._messagePass(nodes1, fmap1, topo1, this.MESSAGE_PASS_ROUNDS);
        fmap2 = this._messagePass(nodes2, fmap2, topo2, this.MESSAGE_PASS_ROUNDS);

        // 5. Cosine similarity with type-compatibility mask
        const sim = this._buildSimilarityMatrix(nodes1, nodes2, fmap1, fmap2, compatGroups, vocab);

        // 6. Greedy strong-first matching (neighbor-constrained)
        const { assignments, matched1, matched2 } = this._greedyMatch(nodes1, nodes2, sim, this.SOFT_MATCH_THRESHOLD, topo1, topo2);

        // 7. Local graph expansion for unmatched nodes
        this._expandMatches(assignments, matched1, matched2, nodes1, nodes2, topo1, topo2, sim, nodes2.length);

        // 8. Build ModelDifferences output
        return this._buildModelDifferences(assignments, nodes1, nodes2);
    }
};

diffAnalyser.ModelNodesDiffAnalyzerEntry = class {
    constructor(nodeID, node1, node2, nodeDifferences) {
        this._nodeID = nodeID;
        this._node1 = node1;
        this._node2 = node2;
        this._nodeDifferences = nodeDifferences;
    }

    get nodeID() {
        return this._nodeID;
    }

    get node1() {
        return this._node1;
    }

    get node2() {
        return this._node2;
    }

    get nodeDifferences() {
        return this._nodeDifferences;
    }

    get generalStatus() {
        return this._nodeDifferences.generalStatus;
    }
};

diffAnalyser.PropertyDifferences = class {
    constructor(name, value1, value2, propertyStatus) {
        this._name = name;
        this._value1 = value1;
        this._value2 = value2;
        this._propertyStatus = propertyStatus;
    }

    get name() {
        return this._name;
    }

    get value1() {
        return this._value1;
    }

    get value2() {
        return this._value2;
    }

    get propertyStatus() {
        return this._propertyStatus;
    }
};

diffAnalyser.PropertiesDifferences = class {
    constructor(propertyDiffs, attributeDiffs, inputDiffs, outputDiffs, metadataDiffs, metricDiffs, generalStatus) {
        this._propertyDiffs = propertyDiffs;
        this._attributeDiffs = attributeDiffs;
        this._inputDiffs = inputDiffs;
        this._outputDiffs = outputDiffs;
        this._metadataDiffs = metadataDiffs;
        this._metricDiffs = metricDiffs;
        this._generalStatus = generalStatus;
    }

    get generalStatus() {
        return this._generalStatus;
    }

    get propertyDiffs() {
        return this._propertyDiffs;
    }

    get attributeDiffs() {
        return this._attributeDiffs;
    }

    get inputDiffs() {
        return this._inputDiffs;
    }

    get outputDiffs() {
        return this._outputDiffs;
    }

    get metadataDiffs() {
        return this._metadataDiffs;
    }

    get metricDiffs() {
        return this._metricDiffs;
    }
};

/* -------------------- Generic Diff Analyzer -------------------- */

diffAnalyser.GenericDiffAnalyzer = class {
    static _getProperties(/* node */) {
        // Override in subclasses
        return [];
    }

    static _getAttributes(genericObject) {
        const attributes = [];
        if (Array.isArray(genericObject?.attributes)) {
            for (const attribute of genericObject.attributes) {
                attributes.push({ name: attribute.name, value: attribute });
            }
        }
        return attributes;
    }

    static _getInputs(genericObject) {
        const inputs = [];
        if (Array.isArray(genericObject?.inputs)) {
            for (const input of genericObject.inputs) {
                inputs.push({ name: input.name, value: input });
            }
        }
        return inputs;
    }

    static _getOutputs(genericObject) {
        const outputs = [];
        if (Array.isArray(genericObject?.outputs)) {
            for (const output of genericObject.outputs) {
                outputs.push({ name: output.name, value: output });
            }
        }
        return outputs;
    }

    static _getMetadata(/* genericObject, model */) {
        // Override in subclasses
        return [];
    }

    static _getMetrics(/* genericObject, model */) {
        // Override in subclasses
        return [];
    }

    static _compareProperties(properties1, properties2) {
        const propertyDiffs = [];
        let generalStatus = DiffStatus.SAME;

        const allPropertyNames = new Set([
            ...properties1.map(p => p.name),
            ...properties2.map(p => p.name)
        ]);

        for (const propertyName of allPropertyNames) {
            const property1 = properties1.find(p => p.name === propertyName);
            const property2 = properties2.find(p => p.name === propertyName);

            const propertyStatus = diffAnalyser.PropertyDiffAnalyzer.compare(property1, property2);

            propertyDiffs.push(
                new diffAnalyser.PropertyDifferences(
                    propertyName,
                    property1 ? property1.value : null,
                    property2 ? property2.value : null,
                    propertyStatus
                )
            );

            // If any property is different/added/removed, the node is considered different.
            if (generalStatus === DiffStatus.SAME) {
                if (propertyStatus.generalStatus !== DiffStatus.SAME) {
                    generalStatus = DiffStatus.DIFF;
                }
            }
        }
        return [propertyDiffs, generalStatus];
    }

    static compare(genericObject1, genericObject2, model1 = null, model2 = null) {
        const properties1 = this._getProperties(genericObject1);
        const properties2 = this._getProperties(genericObject2);
        const [nodePropertyDifferences, propertyStatus] = this._compareProperties(properties1, properties2);

        const attributes1 = this._getAttributes(genericObject1);
        const attributes2 = this._getAttributes(genericObject2);
        const [nodeAttributeDifferences, attributeStatus] = this._compareProperties(attributes1, attributes2);

        const inputs1 = this._getInputs(genericObject1);
        const inputs2 = this._getInputs(genericObject2);
        const [nodeInputDifferences, inputStatus] = this._compareProperties(inputs1, inputs2);

        const outputs1 = this._getOutputs(genericObject1);
        const outputs2 = this._getOutputs(genericObject2);
        const [nodeOutputDifferences, outputStatus] = this._compareProperties(outputs1, outputs2);

        let nodeMetadataDifferences = [];
        let metadataStatus = DiffStatus.SAME;
        if (model1 && model2) {
            const metadata1 = this._getMetadata(genericObject1, model1);
            const metadata2 = this._getMetadata(genericObject2, model2);
            [nodeMetadataDifferences, metadataStatus] = this._compareProperties(metadata1, metadata2);
        }

        let nodeMetricDifferences = [];
        let metricStatus = DiffStatus.SAME;
        if (model1 && model2) {
            const metrics1 = this._getMetrics(genericObject1, model1);
            const metrics2 = this._getMetrics(genericObject2, model2);
            [nodeMetricDifferences, metricStatus] = this._compareProperties(metrics1, metrics2);
        }

        // If any group is different/added/removed, the objects are considered different.
        const generalStatus = [propertyStatus, attributeStatus, inputStatus, outputStatus, metadataStatus, metricStatus]
            .some(status => [DiffStatus.DIFF, DiffStatus.ADDED, DiffStatus.REMOVED].includes(status))
            ? DiffStatus.DIFF
            : DiffStatus.SAME;

        return new diffAnalyser.PropertiesDifferences(
            nodePropertyDifferences,
            nodeAttributeDifferences,
            nodeInputDifferences,
            nodeOutputDifferences,
            nodeMetadataDifferences,
            nodeMetricDifferences,
            generalStatus
        );
    }
};

/* -------------------- Specialized Diff Analyzers -------------------- */

diffAnalyser.NodeDiffAnalyzer = class extends diffAnalyser.GenericDiffAnalyzer {

    static _getProperties(node) {
        const properties = [];
        if (node.type) {
            const typeName = node.type.identifier || node.type.name || '';
            properties.push({ name: 'type', value: typeName });

            if (node.type.module || node.type.version || node.type.status) {
                const list = [
                    node.type.module ? node.type.module : '',
                    node.type.version ? `v${node.type.version}` : '',
                    node.type.status ? node.type.status : ''
                ];
                const value = list.filter(Boolean).join(' ');
                properties.push({ name: 'module', value });
            }
        }

        if (node.name) {
            properties.push({ name: 'name', value: node.name });
        }

        if (node.identifier) {
            properties.push({ name: 'identifier', value: node.identifier });
        }

        if (node.description) {
            properties.push({ name: 'description', value: node.description });
        }

        if (node.device) {
            properties.push({ name: 'device', value: node.device });
        }
        return properties;
    }

    static _getMetadata(node, model) {
        const out = [];
        const metadataList = model?.attachment?.metadata?.node?.(node);
        if (Array.isArray(metadataList) && metadataList.length > 0) {
            for (const argument of metadataList) {
                out.push({ name: argument.name, value: argument });
            }
        }
        return out;
    }

    static _getMetrics(node, model) {
        const out = [];
        const metricsList = model?.attachment?.metrics?.node?.(node);
        if (Array.isArray(metricsList) && metricsList.length > 0) {
            for (const argument of metricsList) {
                out.push({ name: argument.name, value: argument });
            }
        }
        return out;
    }
};

diffAnalyser.GraphDiffAnalyzer = class extends diffAnalyser.GenericDiffAnalyzer {

    static _getProperties(targetSignature) {
        const target = targetSignature[0];
        const signature = targetSignature[1];
        const properties = [];

        if (target?.name) {
            properties.push({ name: 'name', value: target.name });
        }

        if (signature?.name) {
            properties.push({ name: 'signature', value: signature.name });
        }

        if (target?.version) {
            properties.push({ name: 'version', value: target.version });
        }

        if (target?.description) {
            properties.push({ name: 'description', value: target.description });
        }

        return properties;
    }

    static _getAttributes(targetSignature) {
        const target = targetSignature[0];
        const signature = targetSignature[1];
        const attributesList = signature ? signature.attributes : target?.attributes;
        const attributes = [];
        if (Array.isArray(attributesList)) {
            for (const attribute of attributesList) {
                attributes.push({ name: attribute.name, value: attribute });
            }
        }
        return attributes;
    }

    static _getInputs(targetSignature) {
        const target = targetSignature[0];
        const signature = targetSignature[1];
        const inputsList = signature ? signature.inputs : target?.inputs;
        const inputs = [];
        if (Array.isArray(inputsList)) {
            for (const input of inputsList) {
                inputs.push({ name: input.name, value: input });
            }
        }
        return inputs;
    }

    static _getOutputs(targetSignature) {
        const target = targetSignature[0];
        const signature = targetSignature[1];
        const outputsList = signature ? signature.outputs : target?.outputs;
        const outputs = [];
        if (Array.isArray(outputsList)) {
            for (const output of outputsList) {
                outputs.push({ name: output.name, value: output });
            }
        }
        return outputs;
    }

    static _getMetadata(targetSignature, model) {
        const target = targetSignature[0];
        const out = [];
        const metadataList = model?.attachment?.metadata?.graph?.(target);
        if (Array.isArray(metadataList) && metadataList.length > 0) {
            for (const argument of metadataList) {
                out.push({ name: argument.name, value: argument });
            }
        }
        return out;
    }

    static _getMetrics(targetSignature, model) {
        const target = new metrics.Target(targetSignature[0]);
        const out = [];
        const metricsList = model?.attachment?.metrics?.graph?.(target);
        if (Array.isArray(metricsList) && metricsList.length > 0) {
            for (const argument of metricsList) {
                out.push({ name: argument.name, value: argument });
            }
        }
        return out;
    }
};

diffAnalyser.ModelDiffAnalyzer = class extends diffAnalyser.GenericDiffAnalyzer {

    static _getProperties(model) {
        const properties = [];

        if (model.format) {
            properties.push({ name: 'format', value: model.format });
        }
        if (model.producer) {
            properties.push({ name: 'producer', value: model.producer });
        }
        if (model.name) {
            properties.push({ name: 'name', value: model.name });
        }
        if (model.version) {
            properties.push({ name: 'version', value: model.version });
        }
        if (model.description) {
            properties.push({ name: 'description', value: model.description });
        }
        if (model.domain) {
            properties.push({ name: 'domain', value: model.domain });
        }
        if (model.imports) {
            properties.push({ name: 'imports', value: model.imports });
        }
        if (model.runtime) {
            properties.push({ name: 'runtime', value: model.runtime });
        }
        if (model.source) {
            properties.push({ name: 'source', value: model.source });
        }
        return properties;
    }

    static _getMetadata(model /*, same_model */) {
        const out = [];
        const metadataList = model?.attachment?.metadata?.model?.(model);
        if (Array.isArray(metadataList) && metadataList.length > 0) {
            for (const argument of metadataList) {
                out.push({ name: argument.name, value: argument });
            }
        }
        return out;
    }

    static _getMetrics(model /*, same_model */) {
        const modelMetrics = new metrics.Model(model);
        const out = [];
        const metricsList = model?.attachment?.metrics?.model?.(modelMetrics);
        if (Array.isArray(metricsList) && metricsList.length > 0) {
            for (const argument of metricsList) {
                out.push({ name: argument.name, value: argument });
            }
        }
        return out;
    }
};


diffAnalyser.PropertyDiffAnalyzer = class {
    // -------- Helpers --------

    static _isNullOrUndefined(item) {
        return item === null || item === undefined;
    }

    static _isPrimitive(item) {
        return (
            typeof item === 'string' ||
            typeof item === 'number' ||
            typeof item === 'boolean' ||
            typeof item === 'bigint'
        );
    }

    static _hasOwn(obj, key) {
        // Safe own-property check (no shadowing)
        return Object.hasOwn ? Object.hasOwn(obj, key) : Object.prototype.hasOwnProperty.call(obj, key);
    }

    static _isTensor(x) {
        if (x instanceof keras.Tensor) {
            return true;
        }
        else if (x instanceof pytorch.Tensor) {
            return true;
        }
        return false;
    }

    static _unwrapIfWrapper(x) {
        const isObj = x !== null && typeof x === 'object';
        const looksWrapped = isObj && (this._hasOwn(x, 'value') || this._hasOwn(x, 'type'));
        if (looksWrapped) {
            return { value: x.value, type: x.type, wrapped: true };
        }
        return { value: x, type: undefined, wrapped: false };
    }

    // -------- Public API --------

    static compare(property1, property2) {
        const isNil = this._isNullOrUndefined;

        // Treat null/undefined as "absence" consistently.
        if (isNil(property1) && isNil(property2)) {
            return new diffAnalyser.PropertyStatus(DiffStatus.SAME);
        } else if (!isNil(property1) && isNil(property2)) {
            return new diffAnalyser.PropertyStatus(DiffStatus.ADDED);
        } else if (isNil(property1) && !isNil(property2)) {
            return new diffAnalyser.PropertyStatus(DiffStatus.REMOVED);
        }

        // Normalize wrappers vs raw values
        const A = this._unwrapIfWrapper(property1);
        const B = this._unwrapIfWrapper(property2);

        const value1 = A.value;
        const value2 = B.value;

        // Apply the same absence semantics at the value level
        if (isNil(value1) && isNil(value2)) {
            return new diffAnalyser.PropertyStatus(DiffStatus.SAME);
        } else if (!isNil(value1) && isNil(value2)) {
            return new diffAnalyser.PropertyStatus(DiffStatus.ADDED);
        } else if (isNil(value1) && !isNil(value2)) {
            return new diffAnalyser.PropertyStatus(DiffStatus.REMOVED);
        }

        // If both are primitives, compare directly
        if (this._isPrimitive(value1) && this._isPrimitive(value2)) {
            return this._primitivePropertiesEquivalent(value1, value2);
        }

        // Type-aware dispatch (prefer the provided type from either side)
        const type = A.type ?? B.type;

        if (type === 'tensor' || type === 'tensor?') {
            // Single tensor comparison (compareValues = true)
            const ps = new diffAnalyser.PropertyStatus();
            this._tensorsAreEquivalent(value1, value2, ps, /* compareValues */ true);
            return ps;
        }

        if (type === 'tensor[]' || type === 'tensor?[]') {
            const ps = new diffAnalyser.PropertyStatus();
            if (!Array.isArray(value1) || !Array.isArray(value2) || value1.length !== value2.length) {
                ps.setGeneralStatus(DiffStatus.DIFF);
                return ps;
            }
            for (let i = 0; i < value1.length; i++) {
                // Delegate to object comparer so tensor specialization is used
                const inner = new diffAnalyser.PropertyStatus();
                this._objectsAreEquivalent(value1[i], value2[i], inner);
                if (inner.generalStatus !== DiffStatus.SAME) {
                    return inner; // propagate diff reason/status
                }
            }
            ps.setGeneralStatus(DiffStatus.SAME);
            return ps;
        }

        if (type === 'attribute') {
            return this._attributesAreEquivalent(value1, value2);
        }

        // Generic deep compare (handles objects, arrays, tensors as encountered)
        const propertyStatus = new diffAnalyser.PropertyStatus();
        this._objectsAreEquivalent(value1, value2, propertyStatus);
        return propertyStatus;
    }

    // -------- Primitive compare --------

    static _primitivePropertiesEquivalent(value1, value2) {
        // Treat NaN as equal (optional – disable by removing bothNaN)
        const bothNaN =
            typeof value1 === 'number' &&
            typeof value2 === 'number' &&
            Number.isNaN(value1) &&
            Number.isNaN(value2);

        return value1 === value2 || bothNaN
            ? new diffAnalyser.PropertyStatus(DiffStatus.SAME)
            : new diffAnalyser.PropertyStatus(DiffStatus.DIFF);
    }

    // -------- Tensor compare --------

    static _tensorDatasAreEquivalent(tensorValue1, tensorValue2) {
        // Deep compare of raw value buffers / objects
        return diffAnalyser.PropertyDiffAnalyzer._genericObjectsAreEquivalent(
            tensorValue1,
            tensorValue2
        );
    }

    static _tensorsAreEquivalent(a, b, propertyStatus, compareValues = false) {
        if (propertyStatus.tensorValueStatus === DiffStatus.DIFF )
        {
            // Diff already
            return;
        }
        if (a === b) {
            propertyStatus.setGeneralStatus(DiffStatus.SAME);
            return;    
        }
        if (typeof a !== typeof b) {
            propertyStatus.setGeneralStatus(DiffStatus.DIFF);
            return;    
        }
        if (a === null || b === null) {
            propertyStatus.setGeneralStatus(DiffStatus.DIFF);
            return;    
        }
        if (typeof a !== 'object') {
            propertyStatus.setGeneralStatus(DiffStatus.DIFF);
            return;    
        }

        // Defensive: both must be non-null objects of the same "shape"
        const aKeys = Object.keys(a ?? {});
        const bKeys = Object.keys(b ?? {});
        if (aKeys.length !== bKeys.length) {
            propertyStatus.setGeneralStatus(DiffStatus.DIFF);
            return;
        }

        for (const key of aKeys) {
            if (key === '_data') {
                if (compareValues) {
                    const valuesEqual = this._tensorDatasAreEquivalent(a[key], b[key]);
                    if(valuesEqual){
                        propertyStatus.setTensorValueStatus(DiffStatus.SAME);
                    }
                    else {
                        propertyStatus.setTensorValueStatus(DiffStatus.DIFF);
                    }
                } else {
                    // Values exist but not compared: unknown difference status
                    propertyStatus.setTensorValueStatus(DiffStatus.UNKNOWN);
                    continue;
                }
            }

            if (!this._hasOwn(b, key)) {
                propertyStatus.setGeneralStatus(DiffStatus.DIFF);
                return;
            }

            this._tensorsAreEquivalent(a[key], b[key], propertyStatus, compareValues)
            if (propertyStatus.generalStatus !== DiffStatus.SAME) {
                propertyStatus.setGeneralStatus(DiffStatus.DIFF);
                return;
            }
        }

        propertyStatus.setGeneralStatus(DiffStatus.SAME);
        return;
    }

    // -------- Attribute compare --------
    // Implemented as deep structural equality; adjust to your schema if needed.
    static _attributesAreEquivalent(value1, value2) {
        const equal = diffAnalyser.PropertyDiffAnalyzer._genericObjectsAreEquivalent(value1, value2);
        return new diffAnalyser.PropertyStatus(equal ? DiffStatus.SAME : DiffStatus.DIFF);
    }

    // -------- Generic deep compare producing PropertyStatus --------

    static _objectsAreEquivalent(a, b, propertyStatus = new diffAnalyser.PropertyStatus()) {
        if (a === b) {
            propertyStatus.setGeneralStatus(DiffStatus.SAME);
            return;
        }
        if (typeof a !== typeof b) {
            propertyStatus.setGeneralStatus(DiffStatus.DIFF);
            return;
        }
        if (a === null || b === null) {
            propertyStatus.setGeneralStatus(DiffStatus.DIFF);
            return;
        }
        if (typeof a !== 'object') {
            propertyStatus.setGeneralStatus(DiffStatus.DIFF);
            return;
        }

        // Tensor specialization
        if (diffAnalyser.ModelNodesDiffAnalyzer._isTensorValue(a)) {
            this._tensorsAreEquivalent(a, b, propertyStatus, /* compareValues */ true);
            return;
        }

        // Arrays
        if (Array.isArray(a) && Array.isArray(b)) {
            if (a.length !== b.length) {
                propertyStatus.setGeneralStatus(DiffStatus.DIFF);
                return;
            }
            for (let i = 0; i < a.length; i++) {
                this._objectsAreEquivalent(a[i], b[i], propertyStatus);
                if (propertyStatus.generalStatus !== DiffStatus.SAME) {
                    return;
                }
            }
            propertyStatus.setGeneralStatus(DiffStatus.SAME);
            return;
        }

        // Plain objects
        const aKeys = Object.keys(a);
        const bKeys = Object.keys(b);
        if (aKeys.length !== bKeys.length) {
            propertyStatus.setGeneralStatus(DiffStatus.DIFF);
            return;
        }
        for (const key of aKeys) {
            if (!this._hasOwn(b, key)) {
                propertyStatus.setGeneralStatus(DiffStatus.DIFF);
                return;
            }
            this._objectsAreEquivalent(a[key], b[key], propertyStatus);
            if (propertyStatus.generalStatus !== DiffStatus.SAME) {
                return;
            }
        }
        propertyStatus.setGeneralStatus(DiffStatus.SAME);
        return;
    }

    // -------- Generic deep equality (boolean) --------

    static _genericObjectsAreEquivalent(a, b) {
        if (a === b) return true;
        if (typeof a !== typeof b) return false;
        if (a === null || b === null) return false;
        if (typeof a !== 'object') return false;

        // Arrays
        if (Array.isArray(a) && Array.isArray(b)) {
            if (a.length !== b.length) return false;
            for (let i = 0; i < a.length; i++) {
                if (diffAnalyser.PropertyDiffAnalyzer._genericObjectsAreEquivalent(a[i], b[i]) !== true)
                    return false;
            }
            return true;
        }

        // Plain objects
        const aKeys = Object.keys(a);
        const bKeys = Object.keys(b);
        if (aKeys.length !== bKeys.length) return false;
        for (const key of aKeys) {
            if (!diffAnalyser.PropertyDiffAnalyzer._hasOwn(b, key)) return false;
            if (diffAnalyser.PropertyDiffAnalyzer._genericObjectsAreEquivalent(a[key], b[key]) !== true)
                return false;
        }
        return true;
    }
};

export const ModelNodesDiffAnalyzer = diffAnalyser.ModelNodesDiffAnalyzer;
export const SoftModelNodesDiffAnalyzer = diffAnalyser.SoftModelNodesDiffAnalyzer;
export const GraphDiffAnalyzer = diffAnalyser.GraphDiffAnalyzer;
export const ModelDiffAnalyzer = diffAnalyser.ModelDiffAnalyzer;
export const ModelDifferences = diffAnalyser.ModelDifferences;
