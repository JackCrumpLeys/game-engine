use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::Hash;
use std::marker::PhantomData;
use std::ops::{Deref, Index};

pub struct DirectedGraph<NodeData, EdgeData = ()> {
    nodes: GraphIndexed<Node<NodeData, EdgeData>>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct NodeIndex(pub usize);

pub struct GraphIndexed<T> {
    items: Vec<T>,
}

impl<T> GraphIndexed<T> {
    fn new() -> Self {
        Self { items: Vec::new() }
    }

    fn add(&mut self, item: T) -> NodeIndex {
        let index = self.items.len();
        self.items.push(item);
        NodeIndex(index)
    }

    fn get(&self, index: NodeIndex) -> Option<&T> {
        self.items.get(index.0)
    }

    fn get_mut(&mut self, index: NodeIndex) -> Option<&mut T> {
        self.items.get_mut(index.0)
    }
}

impl<T> Index<NodeIndex> for GraphIndexed<T> {
    type Output = T;

    fn index(&self, index: NodeIndex) -> &Self::Output {
        &self.items[index.0]
    }
}

pub struct Node<NodeData, EdgeData> {
    data: NodeData,
    edges: Vec<Edge<EdgeData>>,
}

pub struct Edge<EdgeData> {
    pub target: NodeIndex,
    pub data: EdgeData,
}

pub struct DfsIter<'a, NodeData, EdgeData, Graph: DirectedGraphOperations<NodeData, EdgeData>> {
    graph: &'a Graph,
    stack: Vec<NodeIndex>,
    visited: HashSet<NodeIndex>,
    _marker: PhantomData<(&'a NodeData, &'a EdgeData)>,
}

impl<'a, NodeData, EdgeData, Graph: DirectedGraphOperations<NodeData, EdgeData>> Iterator
    for DfsIter<'a, NodeData, EdgeData, Graph>
{
    type Item = (NodeIndex, &'a NodeData);

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(idx) = self.stack.pop() {
            if self.visited.contains(&idx) {
                continue;
            }
            self.visited.insert(idx);

            // Get node data
            if let Some(node) = self.graph.entry(idx).into_option() {
                // Add neighbors to stack
                for (.., target) in node.edges() {
                    if !self.visited.contains(&target.node_index()) {
                        self.stack.push(target.node_index());
                    }
                }
                return Some((idx, node.into_data()));
            }
        }
        None
    }
}

impl<NodeData, EdgeData> Default for DirectedGraph<NodeData, EdgeData> {
    fn default() -> Self {
        Self::new()
    }
}

impl<NodeData, EdgeData> DirectedGraph<NodeData, EdgeData> {
    pub fn new() -> Self {
        Self {
            nodes: GraphIndexed::new(),
        }
    }

    pub fn add_node(&mut self, data: NodeData) -> NodeIndex {
        let node = Node {
            data,
            edges: Vec::new(),
        };
        self.nodes.add(node)
    }

    pub fn add_edge(&mut self, from: NodeIndex, to: NodeIndex, data: EdgeData) {
        if let Some(node) = self.nodes.get_mut(from) {
            node.edges.push(Edge { target: to, data });
        }
    }
}

pub struct DirectedGraphIndexMapped<NodeData, NodeIndexType: Hash = NodeIndex, EdgeData = ()> {
    inner: DirectedGraph<NodeData, EdgeData>,
    index_map: HashMap<NodeIndexType, NodeIndex>,
}

impl<NodeData, NodeIndexType: Hash + Eq, EdgeData> Default
    for DirectedGraphIndexMapped<NodeData, NodeIndexType, EdgeData>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<NodeData, NodeIndexType: Hash + Eq, EdgeData>
    DirectedGraphIndexMapped<NodeData, NodeIndexType, EdgeData>
{
    pub fn new() -> Self {
        Self {
            inner: DirectedGraph::new(),
            index_map: HashMap::new(),
        }
    }

    pub fn add_node(&mut self, index: NodeIndexType, data: NodeData) -> NodeIndex {
        let node_index = self.inner.add_node(data);
        self.index_map.insert(index, node_index);
        node_index
    }

    pub fn add_edge_indexed(
        &mut self,
        from: &NodeIndexType,
        to: &NodeIndexType,
        data: EdgeData,
    ) -> Result<(), &'static str> {
        let from_index = self
            .index_map
            .get(from)
            .ok_or("From index not found in graph")?;
        let to_index = self
            .index_map
            .get(to)
            .ok_or("To index not found in graph")?;

        self.inner.add_edge(*from_index, *to_index, data);

        Ok(())
    }
}

impl<NodeData, NodeIndexType: Hash + Eq, EdgeData> Index<&NodeIndexType>
    for DirectedGraphIndexMapped<NodeData, NodeIndexType, EdgeData>
{
    type Output = NodeData;

    fn index(&self, index: &NodeIndexType) -> &Self::Output {
        let node_index = self.index_map.get(index).expect("Index not found in graph");
        self.inner
            .get_data(*node_index)
            .expect("Node data not found")
    }
}

impl<NodeData, NodeIndexType: Hash + Eq, EdgeData> Deref
    for DirectedGraphIndexMapped<NodeData, NodeIndexType, EdgeData>
{
    type Target = DirectedGraph<NodeData, EdgeData>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

pub struct EntryUnknown;
pub struct EntryOccupied;

pub struct NodeEntry<'a, NodeData, EdgeData, Marker = EntryUnknown> {
    node: NodeIndex,
    graph: &'a DirectedGraph<NodeData, EdgeData>,
    _marker: PhantomData<Marker>,
}

impl<'a, NodeData, EdgeData> NodeEntry<'a, NodeData, EdgeData, EntryUnknown> {
    fn new_unknown(node: NodeIndex, graph: &'a DirectedGraph<NodeData, EdgeData>) -> Self {
        NodeEntry {
            node,
            graph,
            _marker: PhantomData,
        }
    }

    pub fn data(&'a self) -> Option<&'a NodeData> {
        self.graph.get_data(self.node)
    }

    pub fn to_data(self) -> Option<&'a NodeData> {
        self.graph.get_data(self.node)
    }

    pub fn assume_occupied(self) -> NodeEntry<'a, NodeData, EdgeData, EntryOccupied> {
        NodeEntry {
            node: self.node,
            graph: self.graph,
            _marker: PhantomData,
        }
    }

    fn into_option(self) -> Option<NodeEntry<'a, NodeData, EdgeData, EntryOccupied>> {
        if self.graph.get_data(self.node).is_some() {
            Some(NodeEntry::new_occupied(self.node, self.graph))
        } else {
            None
        }
    }
}

impl<'a, NodeData, EdgeData> NodeEntry<'a, NodeData, EdgeData, EntryOccupied> {
    fn new_occupied(node: NodeIndex, graph: &'a DirectedGraph<NodeData, EdgeData>) -> Self {
        NodeEntry {
            node,
            graph,
            _marker: PhantomData,
        }
    }

    pub fn data(&'a self) -> &'a NodeData {
        self.graph
            .get_data(self.node)
            .expect("Node data should exist for occupied entry")
    }

    pub fn into_data(self) -> &'a NodeData {
        self.graph
            .get_data(self.node)
            .expect("Node data should exist for occupied entry")
    }
}

impl<'a, NodeData, EdgeData, Marker> NodeEntry<'a, NodeData, EdgeData, Marker> {
    fn node_index(&self) -> NodeIndex {
        self.node
    }

    fn into_node_index(self) -> NodeIndex {
        self.node
    }

    pub fn edges(
        &self,
    ) -> impl Iterator<Item = (&Edge<EdgeData>, NodeEntry<'_, NodeData, EdgeData>)> + '_ {
        let base_iter = if let Some(data) = self.graph.nodes.get(self.node) {
            data.edges.iter()
        } else {
            [].iter()
        };

        base_iter.map(move |edge| {
            let target_entry = NodeEntry {
                node: edge.target,
                graph: self.graph,
                _marker: PhantomData,
            };
            (edge, target_entry)
        })
    }
}

/// Trait that defines useful operations on directed graphs
pub trait DirectedGraphOperations<NodeData, EdgeData>:
    GeneralGraphOperations<NodeData, EdgeData>
{
    fn dfs(&self, start: NodeIndex) -> DfsIter<'_, NodeData, EdgeData, Self>
    where
        Self: std::marker::Sized,
    {
        DfsIter {
            graph: self,
            stack: vec![start],
            visited: HashSet::new(),
            _marker: PhantomData,
        }
    }

    fn has_cycle(&self) -> bool
    where
        Self: Sized,
    {
        #[derive(Clone, PartialEq)]
        enum NodeState {
            Visiting,
            Visited,
            Unvisited,
        }
        let mut stack = Vec::new();
        let mut state = vec![NodeState::Unvisited; self.len()];

        for i in 0..self.len() {
            if state[i] != NodeState::Unvisited {
                continue;
            }

            let idx = NodeIndex(i);
            stack.push(idx);

            while let Some(node_idx) = stack.pop() {
                match state[node_idx.0] {
                    NodeState::Visiting => {
                        state[node_idx.0] = NodeState::Visited;
                    }
                    NodeState::Visited => {
                        continue;
                    }
                    NodeState::Unvisited => {
                        state[node_idx.0] = NodeState::Visiting;
                        stack.push(node_idx);

                        if let Some(n) = self.entry(node_idx).into_option() {
                            for target in n.edges().map(|(_, t)| t.node_index()) {
                                match state[target.0] {
                                    NodeState::Visiting => {
                                        return true;
                                    }
                                    NodeState::Unvisited => {
                                        stack.push(target);
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }
                }
            }
        }

        false
    }

    fn get_cycles(&self) -> Vec<Vec<NodeIndex>>
    where
        Self: Sized,
    {
        // Define the state of a node during DFS
        #[derive(Clone, PartialEq, Eq, Debug)]
        enum NodeState {
            Unvisited,
            Visiting, // Currently in the recursion stack (Grey)
            Visited,  // Fully processed (Black)
        }

        // Represents the steps in our simulated recursion stack
        enum Step {
            Enter(NodeIndex), // Start processing a node
            Leave(NodeIndex), // Finish processing a node (backtrack)
        }

        let mut cycles = Vec::new();
        let mut state = vec![NodeState::Unvisited; self.len()];

        // Keeps track of the nodes in the current path to reconstruct the cycle
        let mut path = Vec::new();
        let mut stack = Vec::new();

        // Iterate over all nodes to handle disconnected graph components
        for i in 0..self.len() {
            // If already processed, skip
            if state[i] != NodeState::Unvisited {
                continue;
            }

            // Start DFS for this component
            stack.push(Step::Enter(NodeIndex(i)));

            while let Some(step) = stack.pop() {
                match step {
                    Step::Enter(u) => {
                        let u_idx = u.0;
                        state[u_idx] = NodeState::Visiting;
                        path.push(u);

                        // We schedule "Leave" to happen *after* we process all neighbors.
                        stack.push(Step::Leave(u));

                        // Add neighbors to stack
                        if let Some(n) = self.entry(u).into_option() {
                            for target in n.edges().map(|(_, t)| t.node_index()) {
                                let v_idx = target.0;

                                match state[v_idx] {
                                    NodeState::Visiting => {
                                        // Back-edge detected! The target is already in our current path.
                                        // We reconstruct the cycle from the path.
                                        if let Some(pos) = path.iter().position(|&x| x == target) {
                                            cycles.push(path[pos..].to_vec());
                                        }
                                    }
                                    NodeState::Unvisited => {
                                        // Forward-edge to a new node, process it.
                                        stack.push(Step::Enter(target));
                                    }
                                    NodeState::Visited => {
                                        // Cross-edge to an already finished node, ignore.
                                    }
                                }
                            }
                        }
                    }
                    Step::Leave(u) => {
                        // We are done with this node and its children.
                        // Backtrack: remove from path and mark as permanently visited.
                        path.pop();
                        state[u.0] = NodeState::Visited;
                    }
                }
            }
        }

        cycles
    }

    /// Uses Kahn's algorithm to perform a topological sort with phases.
    /// Either returns a valid topological ordering of the nodes in batch phases,
    /// or a list of cycles if the graph is not a DAG.
    fn phased_topological_sort(&self) -> Result<Vec<Vec<NodeIndex>>, Vec<Vec<NodeIndex>>>
    where
        Self: std::marker::Sized,
    {
        let mut in_degree = vec![0; self.len()];

        for i in 0..self.len() {
            let node_index = NodeIndex(i);
            if let Some(entry) = self.entry(node_index).into_option() {
                for (_, target_entry) in entry.edges() {
                    let target_idx = target_entry.node_index().0;
                    in_degree[target_idx] += 1;
                }
            }
        }

        let mut queue: VecDeque<NodeIndex> = in_degree
            .iter()
            .enumerate()
            .filter_map(|(i, &deg)| if deg == 0 { Some(NodeIndex(i)) } else { None })
            .collect();

        if queue.is_empty() && !self.is_empty() {
            return Err(self.get_cycles());
        }

        let mut result = Vec::new();

        let mut len = 0;
        let mut last_phase_left = queue.len();
        let mut curr_phase = Vec::new();
        let mut next_phase_size = 0;

        while let Some(node_index) = queue.pop_front() {
            curr_phase.push(node_index);
            len += 1;
            for (_, target_entry) in self.entry(node_index).edges() {
                let target_idx = target_entry.node_index().0;
                in_degree[target_idx] -= 1;
                if in_degree[target_idx] == 0 {
                    queue.push_back(NodeIndex(target_idx));
                    next_phase_size += 1;
                }
            }

            last_phase_left -= 1;
            if last_phase_left == 0 {
                if !curr_phase.is_empty() {
                    result.push(curr_phase.clone());
                    curr_phase.clear();
                }
                last_phase_left = next_phase_size;
                next_phase_size = 0;
            }
        }

        if len == self.len() {
            Ok(result)
        } else {
            Err(self.get_cycles())
        }
    }

    /// Checkl if a path exists from `start` to `end` using DFS
    fn path_exists(&self, start: NodeIndex, end: NodeIndex) -> bool
    where
        Self: Sized,
    {
        let mut stack = vec![start];
        let mut visited = HashSet::new();

        while let Some(current) = stack.pop() {
            if current == end {
                return true;
            }
            if visited.contains(&current) {
                continue;
            }
            visited.insert(current);

            if let Some(entry) = self.entry(current).into_option() {
                for (_, target_entry) in entry.edges() {
                    let target_idx = target_entry.node_index();
                    if !visited.contains(&target_idx) {
                        stack.push(target_idx);
                    }
                }
            }
        }

        false
    }

    fn entry<'a>(&'a self, index: NodeIndex) -> NodeEntry<'a, NodeData, EdgeData>;
}

impl<NodeData, EdgeData> DirectedGraphOperations<NodeData, EdgeData>
    for DirectedGraph<NodeData, EdgeData>
{
    fn entry(&self, index: NodeIndex) -> NodeEntry<'_, NodeData, EdgeData> {
        NodeEntry::new_unknown(index, self)
    }
}

impl<NodeData, EdgeData, NodeIndexType: Hash + Eq> DirectedGraphOperations<NodeData, EdgeData>
    for DirectedGraphIndexMapped<NodeData, NodeIndexType, EdgeData>
{
    fn entry(&self, index: NodeIndex) -> NodeEntry<'_, NodeData, EdgeData> {
        self.inner.entry(index)
    }
}

pub trait GeneralGraphOperations<NodeData, EdgeData> {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn clear(&mut self);
    fn iter_nodes<'a>(&'a self) -> impl Iterator<Item = (NodeIndex, &'a NodeData)>
    where
        NodeData: 'a;
    fn nodes(&self) -> Vec<NodeIndex> {
        (0..self.len()).map(NodeIndex).collect()
    }
    fn get_data(&self, index: NodeIndex) -> Option<&NodeData>;
}

impl<NodeData, EdgeData> GeneralGraphOperations<NodeData, EdgeData>
    for DirectedGraph<NodeData, EdgeData>
{
    fn len(&self) -> usize {
        self.nodes.items.len()
    }

    fn clear(&mut self) {
        self.nodes.items.clear();
    }

    fn iter_nodes<'a>(&'a self) -> impl Iterator<Item = (NodeIndex, &'a NodeData)>
    where
        NodeData: 'a,
    {
        self.nodes
            .items
            .iter()
            .enumerate()
            .map(|(i, node)| (NodeIndex(i), &node.data))
    }

    fn get_data(&self, index: NodeIndex) -> Option<&NodeData> {
        self.nodes.get(index).map(|node| &node.data)
    }
}

impl<NodeData, EdgeData, NodeIndexType: Hash + Eq> GeneralGraphOperations<NodeData, EdgeData>
    for DirectedGraphIndexMapped<NodeData, NodeIndexType, EdgeData>
{
    fn len(&self) -> usize {
        self.inner.len()
    }

    fn clear(&mut self) {
        self.inner.clear();
        self.index_map.clear();
    }

    fn iter_nodes<'a>(&'a self) -> impl Iterator<Item = (NodeIndex, &'a NodeData)>
    where
        NodeData: 'a,
    {
        self.inner.iter_nodes()
    }

    fn get_data(&self, index: NodeIndex) -> Option<&NodeData> {
        self.inner.get_data(index)
    }
}
