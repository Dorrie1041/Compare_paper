Published as a conference paper at ICLR 2023
RETHINKING THE EXPRESSIVE POWER OF GNNS VIA
GRAPH BICONNECTIVITY
Bohang Zhang1∗
Liwei Wang1,2
Shengjie Luo1∗
1National Key Laboratory of General Artificial Intelligence,
School of Intelligence Science and Technology, Peking University
2Center for Data Science, Peking University
zhangbohang@pku.edu.cn,
luosj@stu.pku.edu.cn,
Di He1
{wanglw,dihe}@pku.edu.cn
ABSTRACT
Designing expressive Graph Neural Networks (GNNs) is a central topic in learn-
ing graph-structured data. While numerous approaches have been proposed to
improve GNNs in terms of the Weisfeiler-Lehman (WL) test, generally there is
still a lack of deep understanding of what additional power they can systematically
and provably gain. In this paper, we take a fundamentally different perspective to
study the expressive power of GNNs beyond the WL test. Specifically, we intro-
duce a novel class of expressivity metrics via graph biconnectivity and highlight
their importance in both theory and practice. As biconnectivity can be easily cal-
culated using simple algorithms that have linear computational costs, it is natural
to expect that popular GNNs can learn it easily as well. However, after a thorough
review of prior GNN architectures, we surprisingly find that most of them are not
expressive for any of these metrics. The only exception is the ESAN framework
(Bevilacqua et al., 2022), for which we give a theoretical justification of its power.
We proceed to introduce a principled and more efficient approach, called the Gen-
eralized Distance Weisfeiler-Lehman (GD-WL), which is provably expressive for
all biconnectivity metrics. Practically, we show GD-WL can be implemented by a
Transformer-like architecture that preserves expressiveness and enjoys full paral-
lelizability. A set of experiments on both synthetic and real datasets demonstrates
that our approach can consistently outperform prior GNN architectures.
1
INTRODUCTION
Graph neural networks (GNNs) have recently become the dominant approach for graph representa-
tion learning. Among numerous architectures, message-passing neural networks (MPNNs) are ar-
guably the most popular design paradigm and have achieved great success in various fields (Gilmer
et al., 2017; Hamilton et al., 2017; Kipf & Welling, 2017; Veliˇckovi´c et al., 2018). However, one
major drawback of MPNNs lies in the limited expressiveness: as pointed out by Xu et al. (2019);
Morris et al. (2019), they can never be more powerful than the classic 1-dimensional Weisfeiler-
Lehman (1-WL) test in distinguishing non-isomorphic graphs (Weisfeiler & Leman, 1968). This
inspired a variety of works to design provably more powerful GNNs that go beyond the 1-WL test.
One line of subsequent works aimed to propose GNNs that match the higher-order WL variants
(Morris et al., 2019; 2020; Maron et al., 2019c;a; Geerts & Reutter, 2022). While being highly
expressive, such an approach suffers from severe computation/memory costs. Moreover, there
have been concerns about whether the achieved expressiveness is necessary for real-world tasks
(Veliˇckovi´c, 2022). In light of this, other recent works sought to develop new GNN architectures
with improved expressiveness while still keeping the message-passing framework for efficiency
(Bouritsas et al., 2022; Bodnar et al., 2021b;a; Bevilacqua et al., 2022; Wijesinghe & Wang, 2022,
and see Appendix A for more recent advances). However, most of these works mainly justify their
expressiveness by giving toy examples where WL algorithms fail to distinguish, e.g., by focusing on
regular graphs. On the theoretical side, it is quite unclear what additional power they can system-
atically and provably gain. More fundamentally, to the best of our knowledge (see Appendix D.1),
there is still a lack of principled and convincing metrics beyond the WL hierarchy to formally mea-
sure the expressive power and to guide the design of provably better GNN architectures.
∗Equal Contribution.
1
Published as a conference paper at ICLR 2023
(a) Original graph
(b) Block cut-edge tree
(c) Block cut-vertex tree
Figure 1: An illustration of edge-biconnectivity and vertex-biconnectivity. Cut vertices/edges are
outlined in bold red. Gray nodes in (b)/(c) are edge/vertex-biconnected components, respectively.
In this paper, we systematically study the problem of designing expressive GNNs from a novel
perspective of graph biconnectivity. Biconnectivity has long been a central topic in graph theory
(Bollob´as, 1998). It comprises a series of important concepts such as cut vertex (articulation point),
cut edge (bridge), biconnected component, and block cut tree (see Section 2 for formal definitions).
Intuitively, biconnectivity provides a structural description of a graph by decomposing it into disjoint
sub-components and linking them via cut vertices/edges to form a tree structure (cf. Figure 1(b,c)).
As can be seen, biconnectivity purely captures the intrinsic structure of a graph.
The significance of graph biconnectivity can be reflected in various aspects. Firstly, from a theo-
retical point of view, it is a basic graph property and is linked to many fundamental topics in graph
theory, ranging from path-related problems to network flow (Granot & Veinott Jr, 1985) and span-
ning trees (Kapoor & Ramesh, 1995), and is highly relevant to planar graph isomorphism (Hopcroft
& Tarjan, 1972). Secondly, from a practical point of view, cut vertices/edges have substantial values
in many real applications. For example, chemical reactions are highly related to edge-biconnectivity
of the molecule graph, where the breakage of molecular bonds usually occurs at the cut edges and
each biconnected component often remains unchanged after the reaction. As another example, social
networks are related to vertex-biconnectivity, where cut vertices play an important role in linking
between different groups of people (biconnected components). Finally, from a computational point
of view, the problems related to biconnectivity (e.g., finding cut vertices/edges or constructing block
cut trees) can all be efficiently solved using classic algorithms (Tarjan, 1972), with a computation
complexity equal to graph size (which is the same as an MPNN). Therefore, one may naturally ex-
pect that popular GNNs should be able to learn all things related to biconnectivity without difficulty.
Unfortunately, we show this is not the case. After a thorough analysis of four classes of representa-
tive GNN architectures in literature (see Section 3.1), we find that surprisingly, none of them could
even solve the easiest biconnectivity problem: to distinguish whether a graph has cut vertices/edges
or not (corresponding to a graph-level binary classification). As a result, they obviously failed in
the following harder tasks: (i) identifying all cut vertices (a node-level task); (ii) identifying all
cut edges (an edge-level task); (iii) the graph-level task for general biconnectivity problems, e.g.,
distinguishing a pair of graphs that have non-isomorphic block cut trees. This raises the following
question: can we design GNNs with provable expressiveness for biconnectivity problems?
We first give an affirmative answer to the above question. By conducting a deep analysis of the
recently proposed Equivariant Subgraph Aggregation Network (ESAN) (Bevilacqua et al., 2022), we
prove that the DSS-WL algorithm with node marking policy can precisely identify both cut vertices
and cut edges. This provides a new understanding as well as a strong theoretical justification for the
expressive power of DSS-WL and its recent extensions (Frasca et al., 2022). Furthermore, we give
a fine-grained analysis of several key factors in the framework, such as the graph generation policy
and the aggregation scheme, by showing that neither (i) the ego-network policy without marking
nor (ii) a variant of the weaker DS-WL algorithm can identify cut vertices.
However, GNNs designed based on DSS-WL are usually sophisticated and suffer from high com-
putation/memory costs. The main contribution in this paper is then to give a principled and effi-
cient way to design GNNs that are expressive for biconnectivity problems. Targeting this question,
we restart from the classic 1-WL algorithm and figure out a major weakness in distinguishing bi-
connectivity: the lack of distance information between nodes. Indeed, the importance of distance
information is theoretically justified in our proof for analyzing the expressive power of DSS-WL.
To this end, we introduce a novel color refinement framework, formalized as Generalized Distance
Weisfeiler-Lehman (GD-WL), by directly encoding a general distance metric into the WL aggrega-
2
LGCEFHDJINBAMK{G}{H}{A,B,C}{J,K,L,M,N}{D,E,F,I}CFDJ{A,B,C}{D,E,F,I}{J,K,L}{J,M,N}{F,G}{F,H}{I,J}I{C,D}Published as a conference paper at ICLR 2023
Table 1: Summary of theoretical results on the expressive power of different GNN models for various
biconnectivity problems. We also list the time/space complexity (per WL iteration) for each WL
algorithm, where n and m are the number of nodes and edges of a graph, respectively.
Section 4
Section 3.1
MPNN GSN CWN GraphSNN
3-IGN
1-WL SC-WL CWL OS-WL DSS-WL DS-WL SPD-WL GD-WL 2-FWL
Section 3.2
ESAN
Ours
Model
WL variant
Cut vertex
Cut edge
BCVTree
BCETree
Ref. Theorem
Time
Space1
✗
✗
✗
✗
-
✗
✗
✗
✗
3.1
✗
✗
✗
✗
C.12
n+m n+m -
-
n
n
✓
✗
✓
✗
✓
✗
✓
✗
3.2
C.13
n+m n(n+m) n(n+m)
n2
✗
Unknown
Unknown
Unknown
C.16
n
n
✗
✓
✗
✓
4.1
n2
n
✓
✓
✓
✓
4.2, 4.3
n2
n
✓
✓
✓
✓
4.6
n3
n2
tion procedure. We first prove that as a special case, the Shortest Path Distance WL (SPD-WL) is
expressive for all edge-biconnectivity problems, thus providing a novel understanding of its empiri-
cal success. However, it still cannot identify cut vertices. We further suggest an alternative called the
Resistance Distance WL (RD-WL) for vertex-biconnectivity. To sum up, all biconnectivity problems
can be provably solved within our proposed GD-WL framework.
Practical Implementation. The main advantage of GD-WL lies in its simplicity, efficiency and
parallelizability. We show it can be easily implemented using a Transformer-like architecture by
injecting the distance into Multi-head Attention (Vaswani et al., 2017), similar to Ying et al. (2021a).
Importantly, we prove that the resulting Graph Transformer (called Graphormer-GD) is as expressive
as GD-WL. This offers strong theoretical insights into the power and limits of Graph Transformers.
Empirically, we show Graphormer-GD not only achieves perfect accuracy in detecting cut vertices
and cut edges, but also outperforms prior GNN achitectures on popular benchmark datasets.
2 PRELIMINARY
Notations. We use { } to denote sets and use {{ }} to denote multisets. The cardinality of (multi)set
S is denoted as |S|. The index set is denoted as [n] := {1, · · · , n}. Throughout this paper, we
consider simple undirected graphs G = (V, E) with no repeated edges or self-loops. Therefore,
each edge {u, v} ∈ E can be expressed as a set of two elements. For a node u ∈ V, denote its
neighbors as NG(u) := {v ∈ V : {u, v} ∈ E} and denote its degree as degG(u) := |NG(u)|. A
path P = (u0, · · · , ud) is a tuple of nodes satisfying {ui−1, ui} ∈ E for all i ∈ [d], and its length
is denoted as |P | := d. A path P is said to be simple if it does not go through a node more than
once, i.e. ui ̸= uj for i ̸= j. The shortest path distance between two nodes u and v is denoted to be
disG(u, v) := min{|P | : P is a path from u to v}. The induced subgraph with vertex subset S ⊂ V
is defined as G[S] = (S, ES ) where ES := {{u, v} ∈ E : u, v ∈ S}.
We next introduce the concepts of connectivity, vertex-biconnectivity and edge-biconnectivity.
Definition 2.1. (Connectivity) A graph G is connected if for any two nodes u, v ∈ V, there is a
path from u to v. A vertex set S ⊂ V is a connected component of G if G[S] is connected and for
any proper superset T ⊋ S, G[T ] is disconnected. Denote CC(G) as the set of all connected com-
ponents, then CC(G) forms a partition of the vertex set V. Clearly, G is connected iff |CC(G)| = 1.
Definition 2.2. (Biconnectivity) A node v ∈ V is a cut vertex (or articulation point) of G if re-
moving v increases the number of connected components, i.e., |CC(G[V\{v}])| > |CC(G)|. A
graph is vertex-biconnected if it is connected and does not have any cut vertex. A vertex set S ⊂ V
is a vertex-biconnected component of G if G[S] is vertex-biconnected and for any proper super-
set T ⊋ S, G[T ] is not vertex-biconnected. We can similarly define the concepts of cut edge (or
bridge) and edge-biconnected component (we omit them for brevity). Finally, denote BCCV(G)
(resp. BCCE(G)) as the set of all vertex-biconnected (resp. edge-biconnected) components.
Two non-adjacent nodes u, v ∈ V are in the same vertex-biconnected component iff there are two
paths from u to v that do not intersect (except at endpoints). Two nodes u, v are in the same edge-
biconnected component iff there are two paths from u to v that do not share an edge. On the other
hand, if two nodes are in different vertex/edge-biconnected components, any path between them
1The space complexity of WL algorithms may differ from the corresponding GNN models in training, e.g.,
for DS-WL and GD-WL, due to the need to store intermediate results for back-propagation.
3
Published as a conference paper at ICLR 2023
must go through some cut vertex/edge. Therefore, cut vertices/edges can be regarded as “hubs” in
a graph that link different subgraphs into a whole. Furthermore, the link between cut vertices/edges
and biconnected components forms a tree structure, which are called the block cut tree (cf. Figure 1).
Definition 2.3. (Block cut-edge tree) The block cut-edge tree of graph G = (V, E) is defined as
follows: BCETree(G) := (BCCE(G), E E), where
E E := (cid:8){S1, S2} : S1, S2 ∈ BCCE(G), ∃u ∈ S1, v ∈ S2, s.t. {u, v} ∈ E(cid:9) .
Definition 2.4. (Block cut-vertex tree) The block cut-vertex tree of graph G = (V, E) is defined as
follows: BCVTree(G) := (BCCV(G) ∪ V Cut, E V), where V Cut ⊂ V is the set containing all cut
vertices of G and
E V := (cid:8){S, v} : S ∈ BCCV(G), v ∈ V Cut, v ∈ S(cid:9) .
The following theorem shows that all concepts related to biconnectivity can be efficiently computed.
Theorem 2.5. (Tarjan, 1972) The problems related to biconnectivity, including identifying all cut
vertices/edges, finding all biconnected components (BCCV(G) and BCCE(G)), and building block
cut trees (BCVTree(G) and BCETree(G)), can all be solved using the Depth-First Search algo-
rithm, within a computation complexity linear in the graph size, i.e. Θ(|V| + |E|).
Isomorphism and color refinement algorithms. Two graphs G = (VG, EG) and H = (VH , EH )
are isomorphic (denoted as G ≃ H) if there is an isomorphism (bijective mapping) f : VG → VH
such that for any nodes u, v ∈ VG, {u, v} ∈ EG iff {f (u), f (v)} ∈ EH . A color refinement
algorithm is an algorithm that outputs a color mapping χG : VG → C when taking graph G as input,
where C is called the color set. A valid color refinement algorithm must preserve invariance under
isomorphism, i.e., χG(u) = χH (f (u)) for isomorphism f and node u ∈ VG. As a result, it can be
used as a necessary test for graph isomorphism by comparing the multisets {{χG(u) : u ∈ VG}} and
{{χH (u) : u ∈ VH }}, which we call the graph representations. Similarly, χG(u) can be seen as the
node feature of u ∈ VG, and {{χG(u), χG(v)}} corresponds to the edge feature of {u, v} ∈ EG. All
algorithms studied in this paper fit the color refinement framework, and please refer to Appendix B
for a precise description of several representatives (e.g., the classic 1-WL and k-FWL algorithms).
Problem setup. This paper focuses on the following three types of problems with increasing diffi-
culties. Firstly, we say a color refinement algorithm can distinguish whether a graph is vertex/edge-
biconnected, if for any graphs G, H where G is vertex/edge-biconnected but H is not, their graph
representations are different, i.e. {{χG(u) : u ∈ VG}} ̸= {{χH (u) : u ∈ VH }}. Secondly, we say a
color refinement algorithm can identify cut vertices if for any graphs G, H and nodes u ∈ VG, v ∈
VH where u is a cut vertex but v is not, their node features are different, i.e. χG(u) ̸= χH (v).
Similarly, it can identify cut edges if for any {u, v} ∈ EG and {w, x} ∈ EH where {u, v} is a cut
edge but {w, x} is not, their edge features are different, i.e. {{χG(u), χG(v)}} ̸= {{χH (w), χH (x)}}.
Finally, we say a color refinement algorithm can distinguish block cut-vertex/edge trees, if for any
graphs G, H satisfying BCVTree(G) ̸≃ BCVTree(H) (or BCETree(G) ̸≃ BCETree(H)), their
graph representations are different, i.e. {{χG(u) : u ∈ VG}} ̸= {{χH (u) : u ∈ VH }}.
3
INVESTIGATING KNOWN GNN ARCHITECTURES VIA BICONNECTIVITY
In this section, we provide a comprehensive investigation of popular GNN variants in literature,
including the classic MPNNs, Graph Substructure Networks (GSN) (Bouritsas et al., 2022) and its
variant (Barcel´o et al., 2021), GNN with lifting transformations (MPSN and CWN) (Bodnar et al.,
2021b;a), GraphSNN (Wijesinghe & Wang, 2022), and Subgraph GNNs (e.g., Bevilacqua et al.
(2022)). Surprisingly, we find most of these works are not expressive for any biconnectivity prob-
lems listed above. The only exceptions are the ESAN (Bevilacqua et al., 2022) and several variants,
where we give a rigorous justification of their expressive power for both vertex/edge-biconnectivity.
3.1 COUNTEREXAMPLES
1-WL/MPNNs. We first consider the classic 1-WL. We provide two principled class of counterex-
amples which are formally defined in Examples C.9 and C.10, with a few special cases illustrated in
Figure 2. For each pair of graphs in Figure 2, the color of each node is drawn according to the 1-WL
color mapping. It can be seen that the two graph representations are the same. Therefore, 1-WL
cannot distinguish any biconnectivity problem listed in Section 2.
4
Published as a conference paper at ICLR 2023
(a)
(b)
(c)
(d)
Figure 2: Illustration of four representative counterexamples (see Examples C.9 and C.10 for general
definitions). Graphs in the first row have cut vertices (outlined in bold red) and some also have cut
edges (denoted as red lines), while graphs in the second row do not have any cut vertex or cut edge.
Substructure Counting WL/GSN. Bouritsas et al. (2022) developed a principled approach to boost
the expressiveness of MPNNs by incorporating substructure counts into node features or the 1-
WL aggregation procedure. The resulting algorithm, which we call the SC-WL, is detailed in Ap-
pendix B.3. However, we show no matter what sub-structures are used, the corresponding GSN still
cannot solve any biconnectivity problem listed in Section 2. We give a proof in Appendix C.2 for
the general case that allows arbitrary substructures, based on Examples C.9 and C.10. We also point
out that our negative result applies to the similar GNN variant in Barcel´o et al. (2021).
Theorem 3.1. Let H = {H1, · · · , Hk}, Hi = (Vi, Ei) be any set of connected graphs and denote
n = maxi∈[k] |Vi|. Then SC-WL (Appendix B.3) using the substructure set H cannot solve any
vertex/edge-biconnectivity problem listed in Section 2. Moreover, there exist counterexample graphs
whose sizes (both in terms of vertices and edges) are O(n).
GNNs with lifting transformations (MPSN/CWN). Bodnar et al. (2021b;a) considered another
approach to design powerful GNNs by using graph lifting transformations. In a nutshell, these ap-
proaches exploit higher-order graph structures such as cliques and cycles to design new WL aggre-
gation procedures. Unfortunately, we show the resulting algorithms, called the SWL and CWL, still
cannot solve any biconnectivity problem. Please see Appendix C.2 (Proposition C.12) for details.
Other GNN variants. In Appendix C.2, we discuss other recently proposed GNNs, such as Graph-
SNN (Wijesinghe & Wang, 2022), GNN-AK (Zhao et al., 2022), and NGNN (Zhang & Li, 2021).
Due to space limit, we defer the corresponding negative results in Propositions C.13, C.15 and C.16.
3.2 PROVABLE EXPRESSIVENESS OF ESAN AND DSS-WL
We next switch our attention to a new type of GNN framework proposed in Bevilacqua et al. (2022),
called the Equivariant Subgraph Aggregation Networks (ESAN). The central algorithm in EASN is
called the DSS-WL. Given a graph G, DSS-WL first generates a bag of vertex-shared (sub)graphs
Bπ
G = {{G1, · · · , Gm}} according to a graph generation policy π. Then in each iteration t, the
algorithm refines the color of each node v in each subgraph Gi by jointly aggregating its neighboring
colors in the own subgraph and across all subgraphs. The aggregation formula can be written as:
(u) : u ∈ NGi(v)}}, χt−1
(v), {{χt−1
Gi
(v) : i ∈ [m]}}(cid:1) ,
where hash is a perfect hash function. DSS-WL terminates when χt
G induces a stable vertex parti-
tion. In this paper, we consider node-based graph generation policies, for which each subgraph is
associated to a specific node, i.e. Bπ
G = {{Gv : v ∈ V}}. Some popular choices are node deletion
πND, node marking πNM, k-ego-network πEGO(k), and its node marking version πEGOM(k). A full
description of DSS-WL as well as different policies can be found in Appendix B.4 (Algorithm 3).
(v) := hash (cid:0)χt−1
χt
Gi
Gi
G(v) := hash (cid:0){{χt
χt
G (u) : u ∈ NG(v)}}(cid:1) ,
G (v), {{χt−1
(2)
(1)
Gi
A fundamental question regarding DSS-WL is how expressive it is. While a straightforward analysis
shows that DSS-WL is strictly more powerful than 1-WL, an in-depth understanding on what addi-
tional power DSS-WL gains over 1-WL is still limited. The only new result is the very recent work
of Frasca et al. (2022), who showed a 3-WL upper bound for the expressivity of DSS-WL. Yet, such
a result actually gives a limitation of DSS-WL rather than showing its power. Moreover, there is a
large gap between the highly strong 3-WL and the weak 1-WL. In the following, we take a different
perspective and prove that DSS-WL is expressive for both types of biconnectivity problems.
5
Published as a conference paper at ICLR 2023
Theorem 3.2. Let G = (VG, EG) and H = (VH , EH ) be two graphs, and let χG and χH be the
corresponding DSS-WL color mapping with node marking policy. Then the following holds:
• For any two nodes w ∈ VG and x ∈ VH , if χG(w) = χH (x), then w is a cut vertex if and
only if x is a cut vertex.
• For any two edges {w1, w2} ∈ EG and {x1, x2} ∈ EH , if {{χG(w1), χG(w2)}} =
{{χH (x1), χH (x2)}}, then {w1, w2} is a cut edge if and only if {x1, x2} is a cut edge.
The proof of Theorem 3.2 is highly technical and is deferred to Appendix C.3. By using the basic
results derived in Appendix C.1, we conduct a careful analysis of the DSS-WL color mapping and
discover several important properties. They give insights on why DSS-WL can succeed in distin-
guishing biconnectivity, as we will discuss below.
How can DSS-WL distinguish biconnectivity? We find that a crucial advantage of DSS-WL
over the classic 1-WL is that DSS-WL color mapping implicitly encodes distance information (see
Lemma C.19(e) and Corollary C.24). For example, two nodes u ∈ VG, v ∈ VH will have dif-
ferent DSS-WL colors if the distance set {{disG(u, w) : w ∈ VG}} differs from {{disH (v, w) :
w ∈ VH }}. Our proof highlights that distance information plays a vital role in distinguishing edge-
biconnectivity when combining with color refinement algorithms (detailed in Section 4), and it also
helps distinguish vertex-biconnectivity (see the proof of Lemma C.22). Consequently, our analysis
provides a novel understanding and a strong justification for the success of DSS-WL in two aspects:
the graph representation computed by DSS-WL intrinsically encodes distance and biconnectivity
information, both of which are fundamental structural properties of graphs but are lacking in 1-WL.
Discussions on graph generation policies. Note that Theorem 3.2 holds for node marking policy.
In fact, the ability of DSS-WL to encode distance information heavily relies on node marking as
shown in the proof of Lemma C.19. In contrast, we prove that the ego-network policy πEGO(k)
cannot distinguish cut vertices (Proposition C.14), using the counterexample given in Figure 2(c).
Therefore, our result shows an inherent advantage of node marking than the ego-network policy in
distinguishing a class of non-isomorphic graphs, which is raised as an open question in Bevilacqua
et al. (2022, Section 5). It also highlights a theoretical limitation of πEGO(k) compared with its node
marking version πEGOM(k), a subtle difference that may not have received sufficient attention yet.
For example, both the GNN-AK and GNN-AK-ctx architecture (Zhao et al., 2022) cannot solve
vertex-biconnectivity problems since it is similar to πEGO(k) (see Proposition C.15). On the other
hand, the GNN-AK+ does not suffer from such a drawback although it also uses πEGO(k), because
it further adds distance encoding in each subgraph (which is more expressive than node marking).
Discussions on DS-WL. Bevilacqua et al. (2022); Cotta et al. (2021) also considered a weaker ver-
sion of DSS-WL, called the DS-WL, which aggregates the node color in each subgraph without
interaction across different subgraphs (see formula (10)). We show in Proposition C.16 that unfor-
tunately, DS-WL with common node-based policies cannot identify cut vertices when the color of
each node v is defined as its associated subgraph representation Gv. This theoretically reveals the
importance of cross-graph aggregation and justifies the design of DSS-WL. Finally, we point out
that Qian et al. (2022) very recently proposed an extension of DS-WL that adds a final cross-graph
aggregation procedure, for which our negative result may not hold. It may be an interesting direction
to theoretically analyze the expressiveness of this type of DS-WL in future work.
4 GENERALIZED DISTANCE WEISFEILER-LEHMAN TEST
After an extensive review of prior GNN architectures, in this section we would like to formally study
the following problem: can we design a principled and efficient GNN framework with provable ex-
pressiveness for biconnectivity? In fact, while in Section 3.2 we have proved that DSS-WL can
solve biconnectivity problems, it is still far from enough. Firstly, the corresponding GNNs based on
DSS-WL is usually sophisticated due to the complex aggregation formula (1), which inspires us to
study whether simpler architectures exist. More importantly, DSS-WL suffers from high computa-
tional costs in both time and memory. Indeed, it requires Θ(n2) space and Θ(nm) time per iteration
(using policy πNM) to compute node colors for a graph with n nodes and m edges, which is n times
costly than 1-WL. Given the theoretical linear lower bound in Theorem 2.5, one may naturally raise
the question of how to close the gap by developing more efficient color refinement algorithms.
6
Published as a conference paper at ICLR 2023
We approach the problem by rethinking the classic 1-WL test. We argue that a major weakness of
1-WL is that it is agnostic to distance information between nodes, partly because each node can
only “see” its neighbors in aggregation. On the other hand, the DSS-WL color mapping implicitly
encodes distance information as shown in Section 3.2, which inspires us to formally study whether
incorporating distance in the aggregation procedure is crucial for solving biconnectivity problems.
To this end, we introduce a novel color refinement framework which we call Generalized Distance
Weisfeiler-Lehman (GD-WL). The update rule of GD-WL is very simple and can be written as:
G(v) := hash (cid:0){{(dG(v, u), χt−1
χt
G (u)) : u ∈ V}}(cid:1) ,
(3)
where dG can be an arbitrary distance metric. The full algorithm is described in Algorithm 4.
SPD-WL for edge-biconnectivity. As a special case, when choosing the shortest path distance
dG = disG, we obtain an algorithm which we call SPD-WL. It can be equivalently written as
G(v) := hash (cid:0)χt−1
χt
G (v), {{χt−1
G (u) : u ∈ NG(v)}}, {{χt−1
G (u) : disG(v, u) = n − 1}}, {{χt−1
· · · , {{χt−1
G (u) : disG(v, u) = 2}},
G (u) : disG(v, u) = ∞}}(cid:1) .
(4)
From (4) it is clear that SPD-WL is strictly more powerful than 1-WL since it additionally aggre-
gates the k-hop neighbors for all k > 1. There have been several prior works related to SPD-WL,
including using distance encoding as node features (Li et al., 2020) or performing k-hop aggrega-
tion for some small k (see Appendix D.2 for more related works and discussions). Yet, these works
are either purely empirical or provide limited theoretical analysis (e.g., by focusing only on regular
graphs). Instead, we introduce the general and more expressive SPD-WL framework with a rather
different motivation and perform a systematic study on its expressive power. Our key result confirms
that SPD-WL is fully expressive for all edge-biconnectivity problems listed in Section 2.
Theorem 4.1. Let G = (VG, EG) and H = (VH , EH ) be two graphs, and let χG and χH be the
corresponding SPD-WL color mapping. Then the following holds:
• For any two edges {w1, w2} ∈ EG and {x1, x2} ∈ EH , if {{χG(w1), χG(w2)}} =
{{χH (x1), χH (x2)}}, then {w1, w2} is a cut edge if and only if {x1, x2} is a cut edge.
• If {{χG(w) : w ∈ VG}} = {{χH (w) : w ∈ VH }}, then BCETree(G) ≃ BCETree(H).
Theorem 4.1 is highly non-trivial and perhaps surprising at first sight, as it combines three seemingly
unrelated concepts (i.e., SPD, biconnectivity, and the WL test) into a unified conclusion. We give a
proof in Appendix C.4, which separately considers two cases: χG(w1) ̸= χG(w2) and χG(w1) =
χG(w2) (see Figure 2(b,d) for examples). For each case, the key technique in the proof is to construct
an auxiliary graph (Definitions C.26 and C.34) that precisely characterizes the structural relationship
between nodes that have specific colors (see Corollaries C.31 and C.40). Finally, we highlight that
the second item of Theorem 4.1 may be particularly interesting: while distinguishing general non-
isomorphic graphs are known to be hard (Cai et al., 1992; Babai, 2016), we show distinguishing
non-isomorphic graphs with different block cut-edge trees can be much easily solved by SPD-WL.
RD-WL for vertex-biconnectivity. Unfortunately, while SPD-WL is fully expressive for edge-
biconnectivity, it is not expressive for vertex-biconnectivity. We give a simple counterexample in
Figure 2(c), where SPD-WL cannot distinguish the two graphs. Nevertheless, we find that by using
a different distance metric, problems related to vertex-biconnectivity can also be fully solved. We
propose such a choice called the Resistance Distance (RD) (denoted as disR
G). Like SPD, RD is also
a basic metric in graph theory (Doyle & Snell, 1984; Klein & Randi´c, 1993) and has been widely
used to characterize the relationship between nodes (Sanmartın et al., 2022; Velingker et al., 2022).
Formally, the value of disR
G(u, v) is defined to be the effective resistance between nodes u and v
when treating G as an electrical network where each edge corresponds to a resistance of one ohm.
RD has many elegant properties. First, it is a valid metric: indeed, RD is non-negative, semidefinite,
symmetric, and satisfies the triangular inequality (see Appendix E.2). Moreover, similar to SPD,
we also have 0 ≤ disR
G(u, v) = disG(u, v) if G is a tree. In Appendix E.2,
we further show that RD is highly related to the graph Laplacian and can be efficiently calculated.
Theorem 4.2. Let G = (VG, EG) and H = (VH , EH ) be two graphs, and let χG and χH be the
corresponding RD-WL color mapping. Then the following holds:
G(u, v) ≤ n − 1, and disR
• For any two nodes w ∈ VG and x ∈ VH , if χG(w) = χH (x), then w is a cut vertex if and
only if x is a cut vertex.
• If {{χG(w) : w ∈ VG}} = {{χH (w) : w ∈ VH }}, then BCVTree(G) ≃ BCVTree(H).
7
Published as a conference paper at ICLR 2023
The form of Theorem 4.2 exactly parallels Theorem 4.1, which shows that RD-WL is fully expres-
sive for vertex-biconnectivity. We give a proof of Theorem 4.1 in Appendix C.5. In particular, the
proof of the second item is highly technical due to the challenges in analyzing the (complex) struc-
ture of the block cut-vertex tree. It also highlights that distinguishing non-isomorphic graphs that
have different BCVTrees is much easier than the general case.
Combining Theorems 4.1 and 4.2 immediately yields the following corollary, showing that all bi-
connectivity problems can be solved within our proposed GD-WL framework.
Corollary 4.3. When using both SPD and RD (i.e., by setting dG(u, v) := (disG(u, v), disR
G(u, v))),
the corresponding GD-WL is fully expressive for both vertex-biconnectivity and edge-biconnectivity.
Computational cost. The GD-WL framework only needs a complexity of Θ(n) space and Θ(n2)
time per-iteration for a graph of n nodes and m edges, both of which are strictly less than DSS-WL.
In particular, GD-WL has the same space complexity as 1-WL, which can be crucial for large-scale
tasks. On the other hand, one may ask how much computational overhead there is in preprocessing
pairwise distances between nodes. We show in Appendix E that the computational cost can be
trivially upper bounded by O(nm) for SPD and O(n3) for RD. Note that the preprocessing step only
needs to be executed once, and we find that the cost is negligible compared to the GNN architecture.
Practical implementation. One of the main advantages of GD-WL is its high degree of paralleliz-
ability. In particular, we find GD-WL can be easily implemented using a Transformer-like architec-
ture by injecting distance information into Multi-head Attention (Vaswani et al., 2017), similar to
the structural encoding in Graphormer (Ying et al., 2021a). The attention layer can be written as:
Yh = (cid:2)ϕh
1 (D) ⊙ softmax (cid:0)XWh
Q(XWh
K)⊤ + ϕh
2 (D)(cid:1)(cid:3) XWh
V ,
(5)
where X ∈ Rn×d is the input node features of the previous layer, D ∈ Rn×n is the distance matrix
V ∈ Rd×dH are learnable weight matrices of the h-th
such that Duv = dG(u, v), Wh
head, ϕh
2 are elementwise functions applied to D (possibly parameterized), and ⊙ denotes
the elementwise multiplication. The results Yh ∈ Rn×dH across all heads h are then combined and
projected to obtain the final output Y = (cid:80)
O ∈ RdH ×d. We call the resulting
h YhWh
architecture Graphormer-GD, and the full structure of Graphormer-GD is provided in Appendix E.3.
O where Wh
1 and ϕh
K, Wh
Q, Wh
It is easy to see that the mapping from X to Y in (5) is equivariant and simulates the GD-WL
aggregation. Importantly, we have the following expressivity result, which precisely characterizes
the power and limits of Graphormer-GD. We give a proof in Appendix E.3.
Theorem 4.4. Graphormer-GD is at most as powerful as GD-WL in distinguishing non-isomorphic
graphs. Moreover, when choosing proper functions ϕh
2 and using a sufficiently large number
of heads and layers, Graphormer-GD is as powerful as GD-WL.
1 and ϕh
On the expressivity upper bound of GD-WL. To complete the theoretical analysis, we finally
provide an upper bound of the expressive power for our proposed SPD-WL and RD-WL, by studying
the relationship with the standard 2-FWL (3-WL) algorithm.
Theorem 4.5. The 2-FWL algorithm is more powerful than both SPD-WL and RD-WL. Formally,
the 2-FWL color mapping induces a finer vertex partition than that of both SPD-WL and RD-WL.
We give a proof in Appendix C.6. Using Theorem 4.5, we arrive at the concluding corollary:
Corollary 4.6. The 2-FWL is fully expressive for both vertex-biconnectivity and edge-biconnectivity.
5 EXPERIMENTS
In this section, we perform empirical evaluations of our proposed Graphormer-GD. We mainly con-
sider the following two sets of experiments. Firstly, we would like to verify whether Graphormer-
GD can indeed learn biconnectivity-related metrics easily as our theory predicts. Secondly, we
would like to investigate whether GNNs with sufficient expressiveness for biconnectivity can also
help real-world tasks and benefit the generalization performance as well. The code and models will
be made publicly available at https://github.com/lsj2408/Graphormer-GD.
Synthetic tasks. To test the expressive power of GNNs for biconnectivity metrics, we separately
consider two tasks: (i) Cut Vertex Detection and (ii) Cut Edge Detection. Given a GNN model
8
Published as a conference paper at ICLR 2023
that outputs node features, we add a learnable prediction head that takes each node feature (or two
node features corresponding to each edge) as input and predicts whether it is a cut vertex (cut edge)
or not. The evaluation metric for both tasks is the graph-level accuracy, i.e., given a graph, the
model prediction is considered correct only when all the cut vertices/edges are correctly identified.
To make the results convincing, we construct a challenging dataset that comprises various types of
hard graphs, including the regular graphs with cut vertices/edges and also Examples C.9 and C.10
mentioned in Section 3. We also choose several GNN baselines with different levels of expres-
sive power: (i) classic MPNNs (Kipf & Welling, 2017; Veliˇckovi´c et al., 2018; Xu et al., 2019);
(ii) Graph Substructure Network (Bouritsas et al., 2022); (iii) Graphormer (Ying et al., 2021a). The
details of model configurations, dataset, and training procedure are provided in Appendix F.1.
The results are presented in Table 2.
It
can be seen that baseline GNNs can-
not perfectly solve these synthetic tasks.
In contrast, the Graphormer-GD achieves
100% accuracy on both tasks, implying
that it can easily learn biconnectivity met-
rics even in very difficult graphs. More-
over, while using only SPD suffices to
identify cut edges, it is still necessary to
further incorporate RD to identify cut ver-
tices. This is consistent with our theoreti-
cal results in Theorems 4.1, 4.2 and 4.4.
Table 2: Accuracy on cut vertex (articulation point) and
cut edge (bridge) detection tasks.
Model
Cut Vertex
Detection
Cut Edge
Detection
GCN (Kipf & Welling, 2017)
GAT (Veliˇckovi´c et al., 2018)
GIN (Xu et al., 2019)
GSN (Bouritsas et al., 2022)
Graphormer (Ying et al., 2021a)
51.5%±1.3% 62.4%±1.8%
52.0%±1.3% 62.8%±1.9%
53.9%±1.7% 63.1%±2.2%
60.1%±1.9% 70.7%±2.1%
76.4%±2.8% 84.5%±3.3%
Graphormer-GD (ours)
- w/o. Resistance Distance
100%
83.3%±2.7%
100%
100%
Real-world tasks. We further study the empirical performance of our Graphormer-GD on the real-
world benchmark: ZINC from Benchmarking-GNNs (Dwivedi et al., 2020). To show the scalability
of Graphormer-GD, we train our models on both ZINC-Full (consisting of 250K molecular graphs)
and ZINC-Subset (12K selected graphs). We comprehensively compare our model with prior ex-
pressive GNNs that have been publicly released. For a fair comparison, we ensure that the parameter
budget of both Graphormer-GD and other compared models are around 500K, following Dwivedi
et al. (2020). Details of baselines and settings are presented in Appendix F.2.
The results are shown in Table 3, where our score is averaged over four experiments with differ-
ent seeds. We also list the per-epoch training time of different models on ZINC-subset as well
as their model parameters. It can be seen that Graphormer-GD surpasses or matches all compet-
itive baselines on the test set of both ZINC-Subset and ZINC-Full. Furthermore, we find that the
empirical performance of compared models align with their expressive power measured by graph
biconnectivity. For example, Subgraph GNNs that are expressive for biconnectivity also consistently
outperform classic MPNNs by a large margin. Compared with Subgraph GNNs, the main advan-
tage of Graphormer-GD is that it is simpler to implement, has stronger parallelizability, while still
achieving better performance. Therefore, we believe our proposed architecture is both effective and
efficient and can be well extended to more practical scenarios like drug discovery.
6 CONCLUSION
In this paper, we systematically investigate the expressive power of GNNs via the perspective of
graph biconnectivity. Through the novel lens, we gain strong theoretical insights into the power and
limits of existing popular GNNs. We then introduce the principled GD-WL framework that is fully
expressive for all biconnectivity metrics. We further design the Graphormer-GD architecture that
is provably powerful while enjoying practical efficiency and parallelizability. Experiments on both
synthetic and real-world datasets demonstrate the effectiveness of Graphormer-GD.
There are still many promising directions that have not yet been explored. Firstly, it remains an
important open problem whether biconnectivity can be solved more efficiently in o(n2) time using
equivariant GNNs. Secondly, a deep understanding of GD-WL is generally lacking. For example,
we conjecture that RD-WL can encode graph spectral (Lim et al., 2022) and is strictly more powerful
than SPD-WL in distinguishing general graphs. Thirdly, it may be interesting to further investigate
more expressive distance (structural) encoding schemes beyond RD-WL and explore how to encode
them in Graph Transformers. Finally, one can extend biconnectivity to a hierarchy of higher-order
variants (e.g., tri-connectivity), which provides a completely different view parallel to the WL hier-
archy to study the expressive power and guide designing provably powerful GNNs architectures.
9
Published as a conference paper at ICLR 2023
Table 3: Mean Absolute Error (MAE) on ZINC test set. Following Dwivedi et al. (2020), the
parameter budget of compared models is set to 500k. We use ∗ to indicate the best performance.
Method
Model
Time (s)
Params
Test MAE
ZINC-Subset
ZINC-Full
MPNNs
GIN (Xu et al., 2019)
GraphSAGE (Hamilton et al., 2017)
GAT (Veliˇckovi´c et al., 2018)
GCN (Kipf & Welling, 2017)
MoNet (Monti et al., 2017)
GatedGCN-PE(Bresson & Laurent, 2017)
MPNN(sum) (Gilmer et al., 2017)
PNA (Corso et al., 2020)
Higher-order
GNNs
RingGNN (Chen et al., 2019)
3WLGNN (Maron et al., 2019a)
Substructure-
based GNNs
GSN (Bouritsas et al., 2022)
CIN-Small (Bodnar et al., 2021a)
Subgraph
GNNs
Graph
Transformers
NGNN (Zhang & Li, 2021)
DSS-GNN (Bevilacqua et al., 2022)
GNN-AK (Zhao et al., 2022)
GNN-AK+ (Zhao et al., 2022)
SUN (Frasca et al., 2022)
GT (Dwivedi & Bresson, 2021)
SAN (Kreuzer et al., 2021)
Graphormer (Ying et al., 2021a)
URPE (Luo et al., 2022b)
GD-WL
Graphormer-GD (ours)
8.05
6.02
8.28
5.85
7.19
10.74
-
-
178.03
179.35
-
-
-
-
-
-
15.04
-
-
12.26
12.40
12.52
509,549
505,341
531,345
505,079
504,013
505,011
480,805
387,155
527,283
507,603
∼500k
∼100k
∼500k
445,709
∼500k
∼500k
526,489
588,929
508,577
489,321
491,737
502,793
0.526±0.051
0.398±0.002
0.384±0.007
0.367±0.011
0.292±0.006
0.214±0.006
0.145±0.007
0.142±0.010
0.353±0.019
0.303±0.068
0.101±0.010
0.094±0.004
0.111±0.003
0.097±0.006
0.105±0.010
0.091±0.011
0.083±0.003
0.226±0.014
0.139±0.006
0.122±0.006
0.086±0.007
0.081±0.009∗
0.088±0.002
0.126±0.003
0.111±0.002
0.113±0.002
0.090±0.002
-
-
-
-
-
-
0.044±0.003
0.029±0.001
-
-
-
-
-
-
0.052±0.005
0.028±0.002
0.025±0.004∗
ACKNOWLEDGMENTS
Bohang Zhang is grateful to Ruichen Li for his great help in discussing and checking several of the
main results in this paper, including Theorems 3.1, 3.2, 4.1 and C.58. In particular, after the initial
submission, Ruichen Li discovered a simpler proof of Lemma C.28 and helped complete the proof of
Theorem C.58. Bohang Zhang would also thank Yiheng Du, Kai Yang amd Ruichen Li for correct-
ing some small mistakes in the proof of Lemmas C.20 and C.45. We also thank all the anonymous
reviewers for the careful reviews and the valuable suggestions. Their help has further enhanced
our work. This work is supported by National Key R&D Program of China (2022ZD0114900) and
National Science Foundation of China (NSFC62276005).
REFERENCES
Ralph Abboud, ˙Ismail ˙Ilkan Ceylan, Martin Grohe, and Thomas Lukasiewicz. The surprising power
of graph neural networks with random node initialization. In Proceedings of the Thirtieth Inter-
national Joint Conference on Artificial Intelligence, IJCAI-21, pp. 2112–2118, 2021.
Ralph Abboud, Radoslav Dimitrov, and Ismail Ilkan Ceylan. Shortest path networks for graph
property prediction. In The First Learning on Graphs Conference, 2022.
Robert Ackland et al. Mapping the us political blogosphere: Are conservative bloggers more promi-
nent? In BlogTalk Downunder 2005 Conference, Sydney. BlogTalk Downunder 2005 Conference,
Sydney, 2005.
Noga Alon, Raphael Yuster, and Uri Zwick. Finding and counting given length cycles. Algorithmica,
17(3):209–223, 1997.
Uri Alon and Eran Yahav. On the bottleneck of graph neural networks and its practical implications.
In International Conference on Learning Representations, 2021.
Vikraman Arvind, Frank Fuhlbr¨uck, Johannes K¨obler, and Oleg Verbitsky. On weisfeiler-leman
invariance: Subgraph counts and related graph properties. Journal of Computer and System Sci-
ences, 113:42–59, 2020.
Waiss Azizian and Marc Lelarge. Expressive power of invariant and equivariant graph neural net-
works. In International Conference on Learning Representations, 2021.
10
Published as a conference paper at ICLR 2023
Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization. arXiv preprint
arXiv:1607.06450, 2016.
L´aszl´o Babai. Graph isomorphism in quasipolynomial time.
In Proceedings of the forty-eighth
annual ACM symposium on Theory of Computing, pp. 684–697, 2016.
Muhammet Balcilar, Pierre H´eroux, Benoit Gauzere, Pascal Vasseur, S´ebastien Adam, and Paul
Honeine. Breaking the limits of message passing graph neural networks. In International Con-
ference on Machine Learning, pp. 599–608. PMLR, 2021.
Pablo Barcel´o, Floris Geerts, Juan Reutter, and Maksimilian Ryschkov. Graph neural networks with
local graph parameters. In Advances in Neural Information Processing Systems, volume 34, pp.
25280–25293, 2021.
Beatrice Bevilacqua, Fabrizio Frasca, Derek Lim, Balasubramaniam Srinivasan, Chen Cai, Gopinath
Balamurugan, Michael M Bronstein, and Haggai Maron. Equivariant subgraph aggregation net-
works. In International Conference on Learning Representations, 2022.
Cristian Bodnar, Fabrizio Frasca, Nina Otter, Yu Guang Wang, Pietro Li`o, Guido Montufar, and
Michael M. Bronstein. Weisfeiler and lehman go cellular: CW networks. In Advances in Neural
Information Processing Systems, volume 34, 2021a.
Cristian Bodnar, Fabrizio Frasca, Yuguang Wang, Nina Otter, Guido F Montufar, Pietro Lio, and
Michael Bronstein. Weisfeiler and lehman go topological: Message passing simplicial networks.
In International Conference on Machine Learning, pp. 1026–1037. PMLR, 2021b.
B´ela Bollob´as. Modern graph theory, volume 184. Springer Science & Business Media, 1998.
Giorgos Bouritsas, Fabrizio Frasca, Stefanos P Zafeiriou, and Michael Bronstein. Improving graph
neural network expressivity via subgraph isomorphism counting. IEEE Transactions on Pattern
Analysis and Machine Intelligence, 2022.
Xavier Bresson and Thomas Laurent.
Residual gated graph convnets.
arXiv preprint
arXiv:1711.07553, 2017.
Jin-Yi Cai, Martin F¨urer, and Neil Immerman. An optimal lower bound on the number of variables
for graph identification. Combinatorica, 12(4):389–410, 1992.
Ashok K Chandra, Prabhakar Raghavan, Walter L Ruzzo, Roman Smolensky, and Prasoon Tiwari.
The electrical resistance of a graph captures its commute and cover times. computational com-
plexity, 6(4):312–340, 1996.
Zhengdao Chen, Soledad Villar, Lei Chen, and Joan Bruna. On the equivalence between graph
isomorphism testing and function approximation with gnns. Advances in neural information
processing systems, 32, 2019.
Zhengdao Chen, Lei Chen, Soledad Villar, and Joan Bruna. Can graph neural networks count sub-
structures? In Proceedings of the 34th International Conference on Neural Information Process-
ing Systems, pp. 10383–10395, 2020.
Gabriele Corso, Luca Cavalleri, Dominique Beaini, Pietro Li`o, and Petar Veliˇckovi´c. Principal
neighbourhood aggregation for graph nets. Advances in Neural Information Processing Systems,
33:13260–13271, 2020.
Leonardo Cotta, Christopher Morris, and Bruno Ribeiro. Reconstruction for powerful graph repre-
sentations. In Advances in Neural Information Processing Systems, volume 34, pp. 1713–1726,
2021.
Pim de Haan, Taco Cohen, and Max Welling. Natural graph networks. In Proceedings of the 34th
International Conference on Neural Information Processing Systems, volume 33, pp. 3636–3646,
2020.
Peter G Doyle and J Laurie Snell. Random walks and electric networks, volume 22. American
Mathematical Soc., 1984.
11
Published as a conference paper at ICLR 2023
Vijay Prakash Dwivedi and Xavier Bresson. A generalization of transformer networks to graphs.
AAAI Workshop on Deep Learning on Graphs: Methods and Applications, 2021.
Vijay Prakash Dwivedi, Chaitanya K Joshi, Thomas Laurent, Yoshua Bengio, and Xavier Bresson.
Benchmarking graph neural networks. arXiv preprint arXiv:2003.00982, 2020.
Or Feldman, Amit Boyarski, Shai Feldman, Dani Kogan, Avi Mendelson, and Chaim Baskin. We-
isfeiler and leman go infinite: Spectral and combinatorial pre-colorings. In ICLR 2022 Workshop
on Geometrical and Topological Representation Learning, 2022.
Jiarui Feng, Yixin Chen, Fuhai Li, Anindya Sarkar, and Muhan Zhang. How powerful are k-hop
message passing graph neural networks. arXiv preprint arXiv:2205.13328, 2022.
Robert W Floyd. Algorithm 97: shortest path. Communications of the ACM, 5(6):345, 1962.
Fabrizio Frasca, Beatrice Bevilacqua, Michael Bronstein, and Haggai Maron. Understanding and
extending subgraph gnns by rethinking their symmetries. arXiv preprint arXiv:2206.11140, 2022.
Vikas Garg, Stefanie Jegelka, and Tommi Jaakkola. Generalization and representational limits
In International Conference on Machine Learning, pp. 3419–3430.
of graph neural networks.
PMLR, 2020.
Floris Geerts and Juan L Reutter. Expressiveness and approximation properties of graph neural
networks. In International Conference on Learning Representations, 2022.
Justin Gilmer, Samuel S Schoenholz, Patrick F Riley, Oriol Vinyals, and George E Dahl. Neural
message passing for quantum chemistry. In International conference on machine learning, pp.
1263–1272. PMLR, 2017.
Frieda Granot and Arthur F Veinott Jr. Substitutes, complements and ripples in network flows.
Mathematics of Operations Research, 10(3):471–497, 1985.
Ivan Gutman and W Xiao. Generalized inverse of the laplacian matrix and some applications. Bul-
letin (Acad´emie serbe des sciences et des arts. Classe des sciences math´ematiques et naturelles.
Sciences math´ematiques), pp. 15–23, 2004.
William L Hamilton, Rex Ying, and Jure Leskovec.
Inductive representation learning on large
graphs. In Proceedings of the 31st International Conference on Neural Information Processing
Systems, volume 30, pp. 1025–1035, 2017.
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recog-
nition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp.
770–778, 2016.
John E Hopcroft and Robert Endre Tarjan. Isomorphism of planar graphs. In Complexity of computer
computations, pp. 131–152. Springer, 1972.
Max Horn, Edward De Brouwer, Michael Moor, Yves Moreau, Bastian Rieck, and Karsten Borg-
wardt. Topological graph neural networks. In International Conference on Learning Representa-
tions, 2022.
Yinan Huang, Xingang Peng, Jianzhu Ma, and Muhan Zhang. Boosting the cycle counting power
of graph neural networks with i$ˆ2$-GNNs. In International Conference on Learning Represen-
tations, 2023.
Neil Immerman and Eric Lander. Describing graphs: A first-order approach to graph canonization.
In Complexity theory retrospective, pp. 59–81. Springer, 1990.
Sanjiv Kapoor and Hariharan Ramesh. Algorithms for enumerating all spanning trees of undirected
and weighted graphs. SIAM Journal on Computing, 24(2):247–265, 1995.
Nicolas Keriven and Gabriel Peyr´e. Universal invariant and equivariant graph neural networks. In
Proceedings of the 33rd International Conference on Neural Information Processing Systems, pp.
7092–7101, 2019.
12
Published as a conference paper at ICLR 2023
Sandra Kiefer. Power and limits of the Weisfeiler-Leman algorithm. PhD thesis, Dissertation, RWTH
Aachen University, 2020.
Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint
arXiv:1412.6980, 2014.
Thomas N. Kipf and Max Welling. Semi-supervised classification with graph convolutional net-
works. In International Conference on Learning Representations, 2017.
Douglas J Klein and Milan Randi´c. Resistance distance. Journal of mathematical chemistry, 12(1):
81–95, 1993.
Devin Kreuzer, Dominique Beaini, Will Hamilton, Vincent L´etourneau, and Prudencio Tossou. Re-
thinking graph transformers with spectral attention. In Advances in Neural Information Process-
ing Systems, volume 34, 2021.
Pan Li, Yanbang Wang, Hongwei Wang, and Jure Leskovec. Distance encoding: design provably
In Proceedings of the 34th
more powerful neural networks for graph representation learning.
International Conference on Neural Information Processing Systems, pp. 4465–4478, 2020.
Derek Lim, Joshua Robinson, Lingxiao Zhao, Tess Smidt, Suvrit Sra, Haggai Maron, and Stefanie
Jegelka. Sign and basis invariant networks for spectral graph representation learning. arXiv
preprint arXiv:2202.13013, 2022.
Andreas Loukas. What graph neural networks cannot learn: depth vs width. In International Con-
ference on Learning Representations, 2020.
Shengjie Luo, Tianlang Chen, Yixian Xu, Shuxin Zheng, Tie-Yan Liu, Liwei Wang, and Di He.
One transformer can understand both 2d & 3d molecular data. arXiv preprint arXiv:2210.01765,
2022a.
Shengjie Luo, Shanda Li, Shuxin Zheng, Tie-Yan Liu, Liwei Wang, and Di He. Your transformer
may not be as powerful as you expect. arXiv preprint arXiv:2205.13401, 2022b.
Haggai Maron, Heli Ben-Hamu, Hadar Serviansky, and Yaron Lipman. Provably powerful graph
In Advances in neural information processing systems, volume 32, pp. 2156–2167,
networks.
2019a.
Haggai Maron, Heli Ben-Hamu, Nadav Shamir, and Yaron Lipman. Invariant and equivariant graph
networks. In International Conference on Learning Representations, 2019b.
Haggai Maron, Ethan Fetaya, Nimrod Segol, and Yaron Lipman. On the universality of invariant
networks. In International conference on machine learning, pp. 4363–4371. PMLR, 2019c.
Federico Monti, Davide Boscaini, Jonathan Masci, Emanuele Rodola, Jan Svoboda, and Michael M
Bronstein. Geometric deep learning on graphs and manifolds using mixture model cnns.
In
Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 5115–5124,
2017.
Christopher Morris, Martin Ritzert, Matthias Fey, William L Hamilton, Jan Eric Lenssen, Gaurav
Rattan, and Martin Grohe. Weisfeiler and leman go neural: Higher-order graph neural networks.
In Proceedings of the AAAI conference on artificial intelligence, volume 33, pp. 4602–4609, 2019.
Christopher Morris, Gaurav Rattan, and Petra Mutzel. Weisfeiler and leman go sparse: towards
scalable higher-order graph embeddings. In Proceedings of the 34th International Conference on
Neural Information Processing Systems, pp. 21824–21840, 2020.
Christopher Morris, Yaron Lipman, Haggai Maron, Bastian Rieck, Nils M Kriege, Martin Grohe,
Matthias Fey, and Karsten Borgwardt. Weisfeiler and leman go machine learning: The story so
far. arXiv preprint arXiv:2112.09992, 2021.
Christopher Morris, Gaurav Rattan, Sandra Kiefer, and Siamak Ravanbakhsh. Speqnets: Sparsity-
aware permutation-equivariant graph networks. In International Conference on Machine Learn-
ing, pp. 16017–16042. PMLR, 2022.
13
Published as a conference paper at ICLR 2023
Ryan Murphy, Balasubramaniam Srinivasan, Vinayak Rao, and Bruno Ribeiro. Relational pooling
In International Conference on Machine Learning, pp. 4663–4673.
for graph representations.
PMLR, 2019.
P´al Andr´as Papp and Roger Wattenhofer. A theoretical comparison of graph neural network exten-
sions. arXiv preprint arXiv:2201.12884, 2022.
P´al Andr´as Papp, Karolis Martinkus, Lukas Faber, and Roger Wattenhofer. Dropgnn: random
In Advances in Neural Infor-
dropouts increase the expressiveness of graph neural networks.
mation Processing Systems, volume 34, pp. 21997–22009, 2021.
Chendi Qian, Gaurav Rattan, Floris Geerts, Christopher Morris, and Mathias Niepert. Ordered
subgraph aggregation networks. arXiv preprint arXiv:2206.11168, 2022.
Leonardo FR Ribeiro, Pedro HP Saverese, and Daniel R Figueiredo. struc2vec: Learning node
representations from structural identity. In Proceedings of the 23rd ACM SIGKDD international
conference on knowledge discovery and data mining, pp. 385–394, 2017.
Enrique Fita Sanmartın, Sebastian Damrich, and Fred Hamprecht. The algebraic path problem for
In International Conference on Machine Learning, pp. 19178–19204. PMLR,
graph metrics.
2022.
Ryoma Sato. A survey on the expressive power of graph neural networks.
arXiv preprint
arXiv:2003.04078, 2020.
Ryoma Sato, Makoto Yamada, and Hisashi Kashima. Approximation ratios of graph neural networks
In Proceedings of the 33rd International Conference on Neural
for combinatorial problems.
Information Processing Systems, pp. 4081–4090, 2019.
Ryoma Sato, Makoto Yamada, and Hisashi Kashima. Random features strengthen graph neural
networks. In Proceedings of the 2021 SIAM International Conference on Data Mining (SDM),
pp. 333–341. SIAM, 2021.
Bernhard Scholkopf, Kah-Kay Sung, Christopher JC Burges, Federico Girosi, Partha Niyogi,
Tomaso Poggio, and Vladimir Vapnik. Comparing support vector machines with gaussian kernels
to radial basis function classifiers. IEEE transactions on Signal Processing, 45(11):2758–2765,
1997.
Yu Shi, Shuxin Zheng, Guolin Ke, Yifei Shen, Jiacheng You, Jiyan He, Shengjie Luo, Chang Liu,
Di He, and Tie-Yan Liu. Benchmarking graphormer on large-scale molecular modeling datasets.
arXiv preprint arXiv:2203.04810, 2022.
Rajat Talak, Siyi Hu, Lisa Peng, and Luca Carlone. Neural trees for learning on graphs. In Advances
in Neural Information Processing Systems, volume 34, pp. 26395–26408, 2021.
Robert Tarjan. Depth-first search and linear graph algorithms. SIAM journal on computing, 1(2):
146–160, 1972.
Erik Thiede, Wenda Zhou, and Risi Kondor. Autobahn: Automorphism-based graph neural nets. In
Advances in Neural Information Processing Systems, volume 34, pp. 29922–29934, 2021.
Jan Toenshoff, Martin Ritzert, Hinrikus Wolf, and Martin Grohe. Graph learning with 1d convolu-
tions on random walks. arXiv preprint arXiv:2102.08786, 2021.
Jake Topping, Francesco Di Giovanni, Benjamin Paul Chamberlain, Xiaowen Dong, and Michael M.
Bronstein. Understanding over-squashing and bottlenecks on graphs via curvature. In Interna-
tional Conference on Learning Representations, 2022.
Edwin R van Dam, Jack H Koolen, and Hajime Tanaka. Distance-regular graphs. arXiv preprint
arXiv:1410.6294, 2014.
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in neural information
processing systems, volume 30, 2017.
14
Published as a conference paper at ICLR 2023
Petar Veliˇckovi´c. Message passing all the way up. arXiv preprint arXiv:2202.11097, 2022.
Petar Veliˇckovi´c, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Li`o, and Yoshua
Bengio. Graph attention networks. In International Conference on Learning Representations,
2018.
Ameya Velingker, Ali Kemal Sinop, Ira Ktena, Petar Veliˇckovi´c, and Sreenivas Gollapudi. Affinity-
aware graph networks. arXiv preprint arXiv:2206.11941, 2022.
Cl´ement Vignac, Andreas Loukas, and Pascal Frossard. Building powerful and equivariant graph
neural networks with structural message-passing. In Proceedings of the 34th International Con-
ference on Neural Information Processing Systems, pp. 14143–14155, 2020.
Boris Weisfeiler and Andrei Leman. The reduction of a graph to canonical form and the algebra
which appears therein. NTI, Series, 2(9):12–16, 1968.
Asiri Wijesinghe and Qing Wang. A new perspective on” how graph neural networks go beyond
weisfeiler-lehman?”. In International Conference on Learning Representations, 2022.
Keyulu Xu, Weihua Hu, Jure Leskovec, and Stefanie Jegelka. How powerful are graph neural
networks? In International Conference on Learning Representations, 2019.
Chengxuan Ying, Tianle Cai, Shengjie Luo, Shuxin Zheng, Guolin Ke, Di He, Yanming Shen, and
Tie-Yan Liu. Do transformers really perform badly for graph representation? Advances in Neural
Information Processing Systems, 34, 2021a.
Chengxuan Ying, Mingqi Yang, Shuxin Zheng, Guolin Ke, Shengjie Luo, Tianle Cai, Chenglin Wu,
Yuxin Wang, Yanming Shen, and Di He. First place solution of kdd cup 2021 ogb large-scale
challenge graph-level track. arXiv preprint arXiv:2106.08279, 2021b.
Jiaxuan You, Jonathan M Gomes-Selman, Rex Ying, and Jure Leskovec.
Identity-aware graph
neural networks. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35,
pp. 10737–10745, 2021.
Raphael Yuster and Uri Zwick. Finding even cycles even faster. SIAM Journal on Discrete Mathe-
matics, 10(2):209–222, 1997.
Muhan Zhang and Pan Li. Nested graph neural networks.
Processing Systems, volume 34, pp. 15734–15747, 2021.
In Advances in Neural Information
Lingxiao Zhao, Wei Jin, Leman Akoglu, and Neil Shah. From stars to subgraphs: Uplifting any gnn
with local structure awareness. In International Conference on Learning Representations, 2022.
15
Published as a conference paper at ICLR 2023
Appendix
Table of Contents
A Recent advances in expressive GNNs
B The Weisfeiler-Lehman Algorithms and Recently Proposed Variants
.
.
.
.
.
.
.
.
.
.
.
.
B.1 1-WL Test
.
B.2 k-FWL Test
.
.
.
.
B.3 WL with Substructure Counting (SC-WL)
B.4 Equivariant Subgraph Aggregation WL (DSS-WL) .
.
B.5 Generalized Distance WL (GD-WL)
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
C Proof of Theorems
.
.
C.1 Properties of color refinement algorithms .
.
.
C.2 Counterexamples
.
.
.
C.3 Proof of Theorem 3.2 .
.
.
C.4 Proof of Theorem 4.1 .
.
.
C.5 Proof of Theorem 4.2 .
.
C.6 Proof of Theorem 4.5 .
.
.
C.7 Regarding distance-regular graphs .
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
D Further Discussions with Prior Works
D.1 Known metrics for measuring the expressive power of GNNs .
.
D.2 GNNs with distance encoding .
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
E Implementation of Generalized Distance Weisfeiler-Lehman
.
.
.
E.1 Preprocessing Shortest Path Distance .
.
E.2 Preprocessing Resistance Distance
.
.
E.3 Transformer-based implementation .
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
F Experimental Details
.
F.1 Synthetic Tasks
.
F.2 Real-world Tasks
F.3 More Tasks
.
.
F.4 Efficiency Evaluation .
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
16
17
18
18
19
19
20
21
22
22
23
26
31
40
45
48
51
51
52
52
53
53
55
56
56
57
58
59
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
Published as a conference paper at ICLR 2023
A RECENT ADVANCES IN EXPRESSIVE GNNS
Since the seminal works of Xu et al. (2019); Morris et al. (2019), extensive studies have devoted to
developing new GNN architectures with better expressiveness beyond the 1-WL test. These works
can be broadly classified into the following categories.
Higher-order GNNs. One straightforward way to design provably more expressive GNNs is in-
spired by the higher-order WL tests (see Appendix B.2). Instead of performing node feature ag-
gregation, these higher-order GNNs calculate a feature vector for each k-tuple of nodes (k ≥ 2)
and perform aggregation between features of different tuples using tensor operations (Morris et al.,
2019; Maron et al., 2019b;c;a; Keriven & Peyr´e, 2019; Azizian & Lelarge, 2021; Geerts & Reutter,
2022). In particular, Maron et al. (2019a) leveraged equivariant matrix multiplication to design net-
work layers that mimic the 2-FWL aggregation. Due to the huge computational cost of higher-order
GNNs, several recent works considered improving efficiency by leveraging the sparse and local na-
ture of graphs and designing a “local” version of the k-WL aggregation, which comes at the cost of
some expressiveness (Morris et al., 2020; 2022). The work of Vignac et al. (2020) can also be seen
as a local 2-order GNN and its expressive power is bounded by 3-IGN (Maron et al., 2019c).
Substructure-based GNNs. Another way to design more expressive GNNs is inspired by studying
the failure cases of 1-WL test. In particular, Chen et al. (2020) pointed out that standard MPNNs
cannot detect/count common substructures such as cycles, cliques, and paths. Based on this finding,
Bouritsas et al. (2022) designed the Graph Substructure Network (GSN) by incorporating substruc-
ture counting into node features using a preprocessing step. Such an approach was later extended
by Barcel´o et al. (2021) based on homomorphism counting. Bodnar et al. (2021b;a); Thiede et al.
(2021); Horn et al. (2022) further developed novel WL aggregation schemes that take into account
these substructures (e.g., cycles or cliques). Toenshoff et al. (2021) considered using random walk
techniques to generate small substructures.
Subgraph GNNs. In fact, the graphs indistinguishable by 1-WL tend to possess a high degree of
symmetry (e.g., see Figure 2). Based on this observation, a variety of recent approaches sought
to break the symmetry by feeding subgraphs into an MPNN. To maintain equivariance, a set of
subgraphs is generated symmetrically from the original graph using predefined policies, and the final
output is aggregated across all subgraphs. There have been several subgraph generation policies in
prior works, such as node deletion (Cotta et al., 2021), edge deletion (Bevilacqua et al., 2022), node
marking (Papp & Wattenhofer, 2022), and ego-networks (Zhao et al., 2022; Zhang & Li, 2021; You
et al., 2021). These works also slightly differ in the aggregation schemes. In particular, Bevilacqua
et al. (2022) developed a unified framework, called ESAN, which includes per-layer aggregation
across subgraphs and thus enjoys better expressiveness. Very recently, Frasca et al. (2022) further
extended the framework based on a more relaxed symmetry analysis and proved an upper bound of
its expressiveness to be 3-WL. Qian et al. (2022) provided a theoretical analysis of how subgraph
GNNs relate to k-FWL and also designed an approach to learn policies.
Non-equivariant GNNs. Perhaps one of the simplest way to break the intrinsic symmetry of 1-WL
aggregation is to use non-equivariant GNNs. Indeed, Loukas (2020) proved that if each node in a
GNN is equipped with a unique identifier, then standard MPNNs can already be Turing universal.
There have been several works that exploit this idea to build powerful GNNs, such as using port
numbering (Sato et al., 2019), relational pooling (Murphy et al., 2019), random features (Sato et al.,
2021; Abboud et al., 2021), or dropout techniques (Papp et al., 2021). However, since the resulting
architectures cannot fully preserve equivariance, the sample complexity required for training and
generalization may not be guaranteed (Garg et al., 2020). Therefore, in this paper we only focus on
analyzing and designing equivariant GNNs.
Other approaches. Wijesinghe & Wang (2022); de Haan et al. (2020) designed novel variants of
MPNNs based on more powerful neighborhood aggregation schemes that are aware of the local
graph structure, rather than simply treating neighboring nodes as a set. Li et al. (2020); Velingker
et al. (2022) incorporated distance encoding into node/edge features to enhance the expressive power
of MPNNs. Balcilar et al. (2021); Feldman et al. (2022) utilized spectral information of graphs to
achieve better expressiveness beyond 1-WL. Talak et al. (2021) proposed the Neural Tree Network
that performs message passing between higher-order subgraphs instead of node-level aggregation.
Finally, for a comprehensive survey on expressive GNNs, we refer readers to Sato (2020) and Morris
et al. (2021).
17
Published as a conference paper at ICLR 2023
B THE WEISFEILER-LEHMAN ALGORITHMS AND RECENTLY PROPOSED
VARIANTS
In this section, we give a precise description on the family of Weisfeiler-Lehman algorithms and
several recently proposed variants that are studied in this paper. We first present the classic 1-WL
algorithm (Weisfeiler & Leman, 1968) and the more advanced k-FWL (Cai et al., 1992; Morris et al.,
2019). Then we present several recently proposed WL variants, including WL with Substructure
Counting (SC-WL) (Bouritsas et al., 2022), Overlap Subgraph WL (OS-WL) (Wijesinghe & Wang,
2022), Equivariant Subgraph Aggregation WL (DSS-WL) (Bevilacqua et al., 2022) and Generalized
Distance WL (GD-WL).
Throughout this section, we assume hash : X → C is an injective hash function that can map
“arbitrary objects” to a color in C where C is an abstract set called the color set. Formally, the
domain X comprises all the objects we are interested in:
• R ⊂ X and C ⊂ X ;
• For any finite multiset M with elements in X , M ∈ X ;
• For any tuple c ∈ X k of finite dimension k ∈ N+, c ∈ X .
B.1
1-WL TEST
Given a graph G = (V, E), the 1-dimensional Weisfeiler-Lehman algorithm (1-WL), also called the
color refinement algorithm, iteratively calculates a color mapping χG from each vertex v ∈ V to a
color χG(v) ∈ C. The pseudo code of 1-WL is presented in Algorithm 1. Intuitively, at the beginning
the color of each vertex is initialized to be the same. Then in each iteration, 1-WL algorithm updates
each vertex color by combining its own color with the neighborhood color multiset using a hash
function. This procedure is repeated for a sufficiently large number of iterations T , e.g. T = |V|.
Algorithm 1: The 1-dimensional Weisfeiler-Lehman Algorithm
: Graph G = (V, E) and the number of iterations T
Input
Output: Color mapping χG : V → C
1 Initialize: Pick a fixed (arbitrary) element c0 ∈ C, and let χ0
2 for t ← 1 to T do
3
for each v ∈ V do
G(v) := hash (cid:0)χt−1
χt
G (v), {{χt−1
G (u) : u ∈ NG(v)}}(cid:1)
4
G(v) := c0 for all v ∈ V
5 Return: χT
G
G
G
v ⇐⇒ χt
G = {(χt
G(u) = χt
defined to be u ∼χt
G induces a partition of the vertex set V with an equivalence
G(v) for u, v ∈ V. We call each equivalence
G)−1(c) := {v ∈ V : χt
G(v) = c}.
G(v) :
At each iteration, the color mapping χt
relation ∼χt
class a color class with an associated color c ∈ C, denoted as (χt
The corresponding partition is then denoted as P t
G)−1(c) : c ∈ Ct
v ∈ V} is the color set containing all the presented colors of vertices in G.
An important observation is that each 1-WL iteration refines the partition P t
P t+1
is finite, there must exist an iteration Tstable < |V| such that P Tstable
. It follows that
G = P Tstable
P t
as
the stable partition induced by the 1-WL algorithm, and denote χG as any stable color mapping
(i.e. by picking any χt
G . The
mapping χG serves as a node feature extractor so that χG(v) is the representation of node v ∈ V.
Correspondingly, the multiset {{χG(v) : v ∈ V}} can serve as the representation of graph G.
G with t ≥ Tstable). We can similarly define the inverse mapping χ−1
G to a finer partition
v. Since the number of vertices |V|
the partition stabilizes. We thus denote PG := P Tstable
G , because for any u, v ∈ V, u ∼χt+1
for all t ≥ Tstable, i.e.
v implies u ∼χt
= P Tstable+1
G
G} where Ct
G := {χt
G
G
G
G
G
The 1-WL algorithm can be used to distinguish whether two graphs G and H are isomorphic, by
comparing their graph representations {{χG(v) : v ∈ V}} and {{χH (v) : v ∈ V}}.
If the two
multisets are not equivalent, then G and H are clearly non-isomorphic. Thus 1-WL is a necessary
condition to test graph isomorphism. Nevertheless, the 1-WL test fails when {{χG(v) : v ∈ V}} =
{{χH (v) : v ∈ V}} but G and H are still non-isomorphic (see Figure 2 for a counterexample). This
motivates the more powerful higher-order WL tests, which are illustrated in the next subsection.
18
Published as a conference paper at ICLR 2023
B.2 k-FWL TEST
In this section, we present a family of algorithms called the k-dimensional Folklore Weisfeiler-
Lehman algorithms (k-FWL). Instead of calculating a node color mapping, k-FWL computes a color
mapping on each k-tuple of nodes. The pseudo code of k-FWL (k ≥ 2) is presented in Algorithm 2.
Algorithm 2: The k-dimensional Folklore Weisfeiler-Lehman Algorithm
Input
Output: Color mapping χG : V k → C
: Graph G = (V, E) and the number of iterations T
1 Initialize: Pick three fixed different elements c0, c1, cnode ∈ C, let χ0
for each v ∈ V k where Av ∈ Ck×k is a matrix with elements
G(v) := hash(vec(Av))
Av
ij =
(cid:40) cnode
c0
c1
if vi = vj
if vi ̸= vj and {vi, vj} /∈ E
if vi ̸= vj and {vi, vj} ∈ E
(6)
2 for t ← 1 to T do
3
for each v ∈ V k do
4
5
G(v) := hash (cid:0)χt−1
G (v), {{(χt−1
χt
where Ni(v, u) = (v1, · · · , vi−1, u, vi+1, · · · , vk)
G (N1(v, u)), · · · , χt−1
G (Nk(v, u))) : u ∈ V}}(cid:1)
6 Return: χT
G
Intuitively, at the beginning, the color of each vertex tuple v encodes the full structure (i.e.
iso-
mophism type) of the subgraph induced by the ordered vertex set {vi : i ∈ [k]}, by hashing the
“adjacency” matrix Av defined in (6). Then in each iteration, k-FWL algorithm updates the color
of each vertex tuple by combining its own color with the “neighborhood” color using a hash func-
tion. Here, the neighborhood of a tuple v is all the tuples that differ v by exactly one element.
These k × |V| neighborhood colors are grouped into a multiset of size |V| where each element is a
k-tuple. Finally, the update procedure is repeated for a sufficiently large number of iterations T , e.g.
T = |V|k.
Simiar to 1-WL, the k-FWL color mapping χt
G induces a partition of the set of vertex k-tuples
V k, and each k-FWL iteration refines the partition of the previous iteration. Since the number of
vertex k-tuples |V|k is finite, there must exist an iteration Tstable < |V|k such that the partition no
longer changes after t ≥ Tstable. We denote the stable color mapping as χG by picking any χt
G with
t ≥ Tstable.
The k-FWL algorithm can be used to distinguish whether two graphs G and H are isomorphic, by
comparing their graph representations {{χG(v) : v ∈ V k}} and {{χH (v) : v ∈ V k}}. It has been
proved that k-FWL is strictly more powerful than 1-WL in distinguishing non-isomorphic graphs,
and (k + 1)-FWL is strictly more powerful than k-FWL for all k ≥ 2 (Cai et al., 1992).
Moreover, the k-FWL algorithm can also be used to extract node representations as with 1-WL. To
do this, we can simply define χG(v) := χG(v, · · · , v) as the vertex color of the k-FWL algorithm
(without abuse of notation), which induces a partition PG over vertex set V. It has been shown that
this partition is finer than the partition induces by 1-WL, and also the vertex partition induced by
(k + 1)-FWL is finer than that of k-FWL (Kiefer, 2020).
B.3 WL WITH SUBSTRUCTURE COUNTING (SC-WL)
Recently, Bouritsas et al. (2022) proposed a variant of the 1-WL algorithm by incorporating the so-
called substructure counting into WL aggregation procedure. This yields a algorithm that is provably
powerful than the original 1-WL test.
To describe the algorithm, we first need the notation of automorphism group. Given a graph H =
(VH , EH ), an automorphism of H is a bijective mapping f : VH → VH such that for any two
vertices u, v ∈ VH , {u, v} ∈ EH ⇐⇒ {f (u), f (v)} ∈ EH . It follows that all automorphisms of H
form a group under function composition, which is called the automorphism group and denoted as
Aut(H).
19
Published as a conference paper at ICLR 2023
The automorphism group Aut(H) yields a partition of the vertex set V, called orbits. Formally,
given a vertex v ∈ VH , define its orbit OrbH (v) = {u ∈ VH : ∃f ∈ Aut(H), f (u) = v}. The
set of all orbits H\ Aut(H) := {OrbH (v) : v ∈ VH } is called the quotient of the automorphism.
Denote dH = |H\ Aut(H)| and denote the elements in H\ Aut(H) as {OV
i=1. We are now
ready to describe the procedure of SC-WL.
H,i}dH
Pre-processing. Depending on the tasks, one first specify a set of (small) connected graphs H =
{H1, · · · , Hk}, which will be used for sub-structure counting in the input graph G. Popular choices
of these small graphs are cycles of different lengths (e.g., triangle or square) and cliques. Given a
graph G = (VG, EG), for each vertex v ∈ VG and each graph H ∈ H, the following quantities are
calculated:
H,i(v) := (cid:8)G[S] : S ⊂ V, G[S] ≃ H, v ∈ S, fG[S]→VH (v) ∈ OV
xV
(7)
where fG[S]→VH is any isomorphism that maps the vertices of graph G[S] to those of graph H.
Intuitively, xV
H,i(v) counts the number of induced subgraphs of G that is isomorphic to H and
contains node v, such that the orbit of v is similar to the orbit OV
H,i. The counts corresponding to
different orbits OV
H,i and different graphs H are finally combined and concatenated into a vector:
i ∈ [dH ]
(cid:9) ,
H,i
xV(v) = [xV
H1
where the dimension of xV(v) is D = (cid:80)
(v)⊤, · · · , xV
Hk
i∈[k] di.
(v)⊤]⊤ ∈ ND
+
(8)
Message Passing. The message passing procedure is similar to Algorithm 1, except that the aggre-
gation formula (Line 4) is replaced by the following update rule:
G(v) := hash (cid:0)χt−1
χt
G (u), xV(u)) : u ∈ NG(v)}}(cid:1)
G (v), xV(v), {{(χt−1
(9)
which incorporates the substructure counts (7, 8). Note that the update rule (9) is slightly simpler
than the original paper (Bouritsas et al., 2022, Section 3.2), but the expressive power of the two
formulations are the same.
Finally, we note that the above procedure counts substructures and calculates features xV for each
vertex of G. One can similarly consider calculating substructure counts for each edge of G, and the
conclusion in this paper (Theorem 3.1) still holds. Please refer to Bouritsas et al. (2022) for more
details on how to calculate edge features.
B.4 EQUIVARIANT SUBGRAPH AGGREGATION WL (DSS-WL)
Recently, Bevilacqua et al. (2022) developd a new type of graph neural networks, called Equivariant
Subgraph Aggregation Networks, as well as a new WL variant named DSS-WL. Given a graph
G = (V, E), DSS-WL first generates a bag of graphs Bπ
G = {{G1, · · · , Gm}} which share the
vertices, i.e. Gi = (V, Ei), but differ in the edge sets Ei. Here π denotes the graph generation policy
which determines the edge set Ei for each graph Gi. The initial coloring χ0
(v) for each node
Gi
v ∈ V in graph Gi is also determined by π and can be different across different nodes and graphs.
In each iteration, the algorithm refines the color of each node by jointly aggregating its neighboring
colors in the own graph and across different graphs. This procedure is repeated for a sufficiently
large iterations T to obtain the stable color mappings χGi and χG. The pseudo code of DSS-WL is
presented in Algorithm 3.
The key component in the DSS-WL algorithm is the graph generation policy π which must maintain
symmetry, i.e., be equivairant under permutation of the vertex set. We list several common choices
below:
• Node marking policy π = πNM. In this policy, we have Bπ
G = {{Gv : v ∈ V}} where
Gv = G, i.e., there are |V| graphs in Bπ
G whose structures are the completely the same.
The difference, however, lies in the initial coloring which marks the special node v in the
following way: χ0
(u) = c0 for other nodes u ̸= v, where c0, c1 ∈ C are
Gv
two different colors.
(v) = c1 and χ0
Gv
• Node deletion policy π = πND. The bag of graphs for this policy is also defined as Bπ
G =
{{Gv : v ∈ V}}, but each graph Gv = (V, Ev) has a different edge set Ev := E\{{v, w} :
w ∈ NG(v)}. Intuitively, it removes all edges that connects to node v and thus makes v an
isolated node. The initial coloring is chosen as a constant χ0
(v) = c0 for all v ∈ V and
Gi
Gi ∈ Bπ
G for some fixed color c0 ∈ C.
20
Published as a conference paper at ICLR 2023
: Graph G = (V, E), the number of iterations T , and graph selection policy π
G = {{Gi}}m
i=1, Gi = (V, Ei) and initial coloring χ0
Gi
for
Algorithm 3: DSS Weisfeiler-Lehman Algorithm
Input
Output: Color mapping χG : V → C
1 Initialize: Generate a bag of graphs Bπ
i ∈ [m] according to policy π
G(v) := hash (cid:0){{χt
2 Let χ0
3 for t ← 1 to T do
4
for each v ∈ V do
Gi
(v) : i ∈ [m]}}(cid:1) for each v ∈ V
5
6
7
for i ← 1 to m do
χt
(v) :=
Gi
hash (cid:0)χt−1
Gi
G(v) := hash (cid:0){{χt
χt
(v), {{χt−1
Gi
(v) : i ∈ [m]}}(cid:1)
Gi
(u) : u ∈ NGi(v)}}, χt−1
G (v), {{χt−1
G (u) : u ∈ NG(v)}}(cid:1)
8 Return: χT
G
• Ego network policy π = πEGO(k). In this policy, we also have Bπ
G = {{Gv : v ∈ V}}, Gv =
(V, Ev). The edge set Ev is defined as Ev := {{u, w} ∈ E : disG(u, v) ≤ k, disG(w, v) ≤
k}, which corresponds to a subgraph containing all the k-hop neighbors of v and isolating
(v) = c0 for all v ∈ V and Gi ∈ Bπ
other nodes. The initial coloring is chosen as χ0
G
Gi
where c0 ∈ C is a constant. One can also consider the ego network policy with marking
π = πEGOM(k), by marking the initial color of the special node v for each Gv.
We note that for all the above policies, |Bπ
G| = |V|. There are other choices such as the edge deletion
policy (Bevilacqua et al., 2022), but we do not discuss them in this paper. A straightforward analysis
yields that DSS-WL with any above policy is strictly powerful than the classic 1-WL algorithm.
Also, node marking policy has been shown to be not less powerful than the node deletion policy
(Papp & Wattenhofer, 2022).
Finally, we highlight that Bevilacqua et al. (2022); Cotta et al. (2021) also proposed a weaker version
of DSS-WL, called the DS-WL algorithm. The difference is that for DS-WL, Lines 6 and 7 in
Algorithm 3 are replaced by a simple 1-WL aggregation:
(v), {{χt−1
Gi
(v) := hash (cid:0)χt−1
Gi
(u) : u ∈ NG(v)}}(cid:1) .
(10)
χt
Gi
However, the original formulation of DS-WL (Bevilacqua et al., 2022) only outputs a graph repre-
sentation {{{{χGi(v) : v ∈ V}} : Gi ∈ Bπ
G}} rather than outputs each node color, which does not
suit the node-level tasks (e.g., finding cut vertices). Nevertheless, there are simple adaptations that
makes DS-WL output a color mapping χG. We will study these adaptations in Appendix C.2 (see
the paragraph above Proposition C.16) and discuss their limitations compared with DSS-WL.
B.5 GENERALIZED DISTANCE WL (GD-WL)
In this paper, we study a new variant of the color refinement algorithm, called the Generalized Dis-
tance WL (GD-WL). The complete algorithm is described below. As a special case, when choosing
dG = disG, the resulting algorithm is called the Shortest Path Distance WL (SPD-WL), which is
strictly powerful than the classic 1-WL.
Algorithm 4: The Genealized Distance Weisfeiler-Lehman Algorithm
Input
Output: Color mapping χG : V → C
: Graph G = (V, E), distance metric dG : V × V → R+, and the number of iterations T
1 Initialize: Pick a fixed (arbitrary) element c0 ∈ C, and let χ0
2 for t ← 1 to T do
3
for each v ∈ V do
G(v) := hash (cid:0){{(dG(v, u), χt−1
χt
G (u)) : u ∈ V}}(cid:1)
4
G(v) := c0 for all v ∈ V
5 Return: χT
G
21
Published as a conference paper at ICLR 2023
C PROOF OF THEOREMS
This section provides all the missing proofs in this paper. For the convenience of reading, we will
restate each theorem before giving a proof.
C.1 PROPERTIES OF COLOR REFINEMENT ALGORITHMS
In this subsection, we first derive several important properties that are shared by a general class
of color refinement algorithms. They will serve as key lemmas in our subsequent proofs. Here,
a general color refinement algorithm takes a graph G = (VG, EG) as input and calculates a color
mapping χG : VG → C. We first define a concept called the WL-condition.
Definition C.1. A color mapping χG : VG → C is said to satisfy the WL-condition if for any two
vertices u, v with the same color (i.e. χG(u) = χG(v)) and any color c ∈ C,
|NG(u) ∩ χ−1
G (c)| = |NG(v) ∩ χ−1
G (c)|,
G is the inverse mapping of χG.
where χ−1
Remark C.2. The WL-condition can be further generalized to handle two graphs. Let χG : VG → C
and χH : VH → C be two color mappings obtained by applying the same color refinement algorithm
for graphs G and H, respectively. χG and χH are said to jointly satisfy the WL-condition, if for any
two vertices u ∈ VG and v ∈ VH with the same color (χG(u) = χH (v)) and any color c ∈ C,
|NG(u) ∩ χ−1
G (c)| = |NH (v) ∩ χ−1
H (c)|.
It clearly implies Definition C.1 by choosing G = H.
It is easy to see that the classic 1-WL algorithm (Algorithm 1) satisfies the WL-condition. In fact,
many of the presented algorithms in this paper satisfy such a condition as we will show below, such
as DSS-WL (Algorithm 3), SPD-WL (Algorithm 4 with dG = disG), and k-FWL (Algorithm 2).
Proposition C.3. Consider the DSS-WL algorithm (Algorithm 4) with arbitrary graph selection
policy π. Let χG and χH be the color mappings for graphs G and H, and let {{χGi : i ∈ [mG]}}
and {{χHi : i ∈ [mH ]}} be the color mapping for subgraphs generated by π. Then,
• χG and χH jointly satisfy the WL-condition;
• χGi and χHj jointly satisfy the WL-condition for any i ∈ [mG] and j ∈ [mH ].
Proof. We first prove the second bullet of Proposition C.3. By definition of the DSS-WL aggregation
procedure (Line 6 in Algorithm 3), χGi(u) = χHi (v) already implies {{χGi (w) : w ∈ NGi(u)}} =
{{χHj (w) : w ∈ NHj (v)}}. Namely, |{w : w ∈ NGi(u) ∩ χ−1
(c)}| = |{w : w ∈ NHj (v) ∩
Gi
χ−1
Hj
(c)}| holds for any c ∈ C.
We then turn to the first bullet. If χG(u) = χH (v), then {{χGi(u) : i ∈ [mG]}} = {{χHj (v) :
j ∈ [mH ]}} (Line 7 in Algorithm 3). Then there exists a pair of indices i ∈ [mG] and j ∈ [mH ]
such that χGi(u) = χHj (v). By definition of the DSS-WL aggregation, it implies {{χG(w) : w ∈
NG(u)}} = {{χH (w) : w ∈ NH (v)}} and concludes the proof.
Proposition C.4. Let χG and χH be two mappings returned by SPD-WL (Algorithm 4 with dG =
disG) for graphs G and H, respectively. Then χG and χH jointly satisfy the WL-condition.
Proof. If χG(u) = χH (v) for some nodes u, v, then by the update rule (Line 4 in Algorithm 4)
{{(disG(u, w), χG(w)) : w ∈ V}} = {{(disG(v, w), χG(w)) : w ∈ V}}.
Since w ∈ NG(u) if and only if disG(u, w) = 1, we have
{{χG(w) : w ∈ NG(u)}} = {{χG(w) : w ∈ NG(v)}}.
Therefore, for any c ∈ C, |{w : w ∈ NG(u) ∩ χ−1
G (c)}| = |{w : w ∈ NG(v) ∩ χ−1
G (c)}|.
Proposition C.5. Let χG and χH be two vertex color mappings returned by the k-FWL algorithm
(k ≥ 2). Then χG and χH jointly satisfy the WL-condition.
22
Published as a conference paper at ICLR 2023
Proof. Let χG(u) = χH (v) for some u ∈ VG and v ∈ VH . By the update formula (Line 4 in
Algorithm 2), {{χG(u, · · · , u, w) : w ∈ VG}} = {{χH (v, · · · , v, w) : w ∈ VH }}. Note that for any
nodes w1 ∈ VG, w2 ∈ VH and any x1 ∈ NG(w1), x2 /∈ NH (w2), one has χG(w1, · · · , w1, x1) ̸=
χH (w2, · · · , w2, x2). This is obtained by the definition of the initialization mapping χ0
G and the
fact that χG refines χ0
G. Consequently, {{χG(u, · · · , u, w) : w ∈ NG(u)}} = {{χG(v, · · · , v, w) :
w ∈ NH (v)}}. Next, we can use the fact that if χG(u, · · · , u, w1) = χG(v, · · · , v, w2) for some
w1, w2 ∈ V, then χG(w1) = χG(w2) (see Lemma C.6). Therefore, {{χG(w) : w ∈ NG(u)}} =
{{χG(w) : w ∈ NH (v)}}, which concludes the proof.
To complete the proof of Proposition C.5, it remains to prove the following lemma:
Lemma C.6. Let χG and χH be color mappings for graphs G and H in the k-FWL algorithm
(k ≥ 2). Denote
cati,j(w, x) := (w, · · · , w
, x, · · · , x
(cid:124) (cid:123)(cid:122) (cid:125)
(cid:125)
j times
(cid:123)(cid:122)
i times
(cid:124)
).
Then for any i ∈ [k − 1] and any nodes u, w ∈ VG, v, x ∈ VH , if χG(catk−i,i(u, w)) =
χH (catk−i,i(v, x)),
then χG(catk−i−1,i+1(u, w)) = χH (catk−i−1,i+1(v, x)). Consequently,
χG(w) = χH (x).
Proof. By the update formula (Line 4 in Algorithm 2), χG(catk−i,i(u, w)) = χH (catk−i,i(v, x))
implies that {{χG(catk−i−1,1,i(u, y, w)) : y ∈ VG}} = {{χH (catk−i−1,1,i(v, y, x)) : y ∈ VH }}.
j ̸= z′
Note that for any j ∈ [k − 1] and any z ∈ V k
j+1, one has
χG(z) ̸= χH (z′). This is obtained by the definition of the initialization mapping χ0
G and the fact
that χG refines χ0
G. Therefore, we have χG(catk−i−1,i+1(u, w)) = χH (catk−i−1,i+1(v, x)), as
desired.
H with zj = zj+1 but z′
G, z′ ∈ V k
Equipped with the concept of WL-condition, we now present several key results. In the following,
let χG : VG → C and χH : VH → C be two color mappings jointly satisfying the WL-condition.
Lemma C.7. Let (v0, · · · , vd) be any path (not necessarily simple) of length d in graph G. Then
for any node u0 ∈ χ−1
H (χG(v0)) in graph H, there exists a path (u0, · · · , ud) of the same length d
starting at u0, such that χH (ui) = χG(vi) holds for all i ∈ [d].
Proof. The proof is based on induction over the path length d. For the base case of d = 1, if the
conclusion does not hold, then there exists two vertices u ∈ VG, v ∈ VH with the same color (i.e.
χG(u) = χH (v)) and a color c = χG(v1) such that NG(u) ∩ χ−1
H (c) = ∅.
This obviously contradicts the WL-condition. For the induction step on the path length d, one can
just split it by two parts (v0, · · · , vd−1) and (vd−1, vd). Separately using induction yields two paths
(u0, · · · , ud−1) and (ud−1, ud) such that χH (ui) = χG(vi) for all i ∈ [d]. By linking the two paths
we have completed the proof.
G (c) ̸= ∅ but NH (v) ∩ χ−1
Finally, let us define the shortest path distance between node u and vertex set S as disG(u, S) :=
minv∈S disG(u, v). The above lemma directly yields the following corollary:
Corollary C.8. For any color c ∈ {χG(w) : w ∈ VG} and any two vertices u ∈ VG, v ∈ VH with
the same color (i.e. χG(u) = χH (v)), disG(u, χ−1
G (c)) = disH (v, χ−1
H (c)).
C.2 COUNTEREXAMPLES
We provides the following two families of counterexamples, which most prior works cannot distin-
guish.
Example C.9. Let G1 = (V, E1) and G2 = (V, E2) be a pair of graphs with n = 2km + 1 nodes
where m, k are two positive integers satisfying mk ≥ 3. Denote V = [n] and define the edge sets as
follows:
E1 = {{i, (i mod 2km) + 1} : i ∈ [2km]} ∪ {{n, i} : i ∈ [2km], i mod m = 0} ,
E2 = {{i, (i mod km) + 1} : i ∈ [km]} ∪ {{i + km, (i mod km) + km + 1} : i ∈ [km]} ∪
{{n, i} : i ∈ [2km], i mod m = 0} .
23
Published as a conference paper at ICLR 2023
See Figure 2(a-c) for an illustration of three cases: (i) m = 2, k = 2; (ii) m = 4, k = 1; (iii)
m = 1, k = 4. It is easy to see that regardless of the chosen of m and k, G1 always has no cut
vertex but G2 do always have a cut vertex with node number n. The case of k = 1 is more special,
for which G2 actually has three cut vertices with node number m, 2m, and n, respectively, and it
even has two cut edges {m, n} and {2m, n} (Figure 2(b)).
Example C.10. Let G1 = (V, E1) and G2 = (V, E2) be a pair of graphs with n = 2m nodes where
m ≥ 3 is an arbitrary integer. Denote V = [n] and define the edge sets as follows:
E1 = {{i, (i mod n) + 1} : i ∈ [n]} ∪ {{m, 2m}} ,
E2 = {{i, (i mod m) + 1} : i ∈ [m]} ∪ {{i + m, (i mod m) + m + 1} : i ∈ [m]} ∪ {{m, 2m}} .
See Figure 2(d) for an illustration of the case n = 8. It is easy to see that G1 does not have any cut
vertex or cut edge, but G2 do have two cut vertices with node number m and 2m, and has a cut edge
{m, 2m}.
Theorem C.11. Let H = {H1, · · · , Hk}, Hi = (Vi, Ei) be any set of connected graphs and denote
nV = maxi∈[k] |Vi|. Then SC-WL (Appendix B.3) using the substructure set H can neither distin-
guish whether a given graph has cut vertices nor distinguish whether it has cut edges. Moreover,
there exist counterexample graphs whose size (both in terms of vertices and edges) is O(nV).
Proof. We would like to prove that SC-WL cannot distinguish both Examples C.9 and C.10 when
nV < m (m is defined in these examples). First note that for both examples, any cycle in both G1
and G2 has a length of at least m. Since the number of nodes in Hi is O(nV), if Hi contains cycles,
it will not occur in both G1 and G2, thus taking no effect in distinguishing the two graphs. As a
result, we can simply assume all graphs in H are trees (connected graphs with no cycles). Below,
we provide a complete proof for Example C.9, which already yields the conclusion that SC-WL can
neither distinguish cut vertices nor cut edges. We omit the proof for Example C.10 since the proof
technique is similar.
(n) = xV
G2
Proof for Example C.9. Let Hi be a tree with less than m vertices where m is defined in Exam-
ple C.9. By symmetry of the two graphs G1 and G2, it suffices to prove the following two types of
(i) for all m < i ≤ 2m, where xV is defined in (8).
equations: xV
G1
We first aim to prove that xV
(v) for v ∈ {m + 1, · · · , 2m}. Consider an induced sub-
G1
graph G1[S] which is isomorphic to Hi and contains node v. Define the set T := {jm : j ∈ [k]}∩S.
For ease of presentation, we define an operation cir(x, a, b) that outputs an integer y in the range of
(a, b] such that y has the same remainder as x (mod b − a). Formally, cir(x, a, b) = y if a < y ≤ b
and x ≡ y (mod b − a).
(n) and xV
G1
(v) = xV
G2
(i) = xV
G2
• If n /∈ S, then it is easy to see that G1[S] is a chain, i.e., no vertices have a degree larger
than 2. We define the following mapping gS : S → [n], such that
if k = 1,
if k ≥ 2.
(cid:26) cir(u, m, 2m)
cir(u, 0, km)
gS (u) =
In this way, the chain G1[S] is mapped to a chain of G2 that contains v. Concretely, denote
gS (S) = {gS (u) : u ∈ S}, then G2[gS (S)] ≃ G1[S] ≃ Hi, and obviously the orbit of v in
G2[gS (S)] matches the orbit of v in G1[S]. See Figure 3(a,b) for an illustration of this case.
• If n ∈ S, then it is easy to see that the set T ̸= ∅. We will similarly construct a mapping
gS : S → [n] that maps S to gS (S) satisfying gS (v) = v, which is defined as follows. For
each u ∈ S\{n}, we find a unique vertex wu in T such that disG1[S](u, wu) is the minimum.
Note that the node wu is well-defined since T ̸= ∅ and any path in G1[S] from u to a node
in T goes through wu. Define


cir(u, m, 2m)
cir(u, 0, m)
cir(u, 0, km)
cir(u, km, 2km)

if k = 1 and wu = wv,
if k = 1 and wu ̸= wv,
if k > 1 and wu ≤ km,
if k > 1 and wu > km.
gS (u) =
We also define gS (n) = n. Such a definition guarantees that for any x1, x2 ∈ S, {x1, x2} ∈
EG1 ⇐⇒ {gS (x1), gS (x2)} ∈ EG2. Therefore, G2[gS (S)] ≃ G1[S] ≃ Hi. Moreover,
observe that gS (u) ≡ u (mod m) always holds, and thus it is easy to see that the orbit of
v in G2[gS (S)] matches the orbit of v in G1[S]. See Figure 3(c,d) for an illustration of this
case.
24
Published as a conference paper at ICLR 2023
(a) n /∈ S, k = 1
(b) n /∈ S, k > 1
(c) n ∈ S, k = 1
(d) n ∈ S, k > 1
Figure 3: Illustration of the proof of Theorem 3.1. The trees G1[S], G2[g(S)] are outlined by orange.
Finally, note that for any two different sets S1 and S2 such that G1[S1] ≃ G1[S2] ≃ Hi, we have
gS1(S1) ̸= gS2(S2), which guarantees that the mapping g : {S ⊂ [n] : G1[S] ≃ Hi, v ∈ S} →
{S ⊂ [n] : G2[S] ≃ Hi, v ∈ S} defined to be g(S) = gS (S) is injective. One can further
check that the mapping g is also surjective, and thus it is bijective. This means xV
(v)
G1
for v ∈ {m, · · · , 2m − 1}. The proof for xV
(n) is almost the same, so we omit it
G1
here. Noting that under classic 1-WL, the colors χG1(v) = χG2 (v) are also the same. Therefore,
adding the features xV(v) does not help distinguish the two graphs. We have finished the proof for
Example C.9.
(n) = xV
G2
(v) = xV
G2
Using a similar cycle analysis as the above proof, we have the following negative result for Simplicial
WL (Bodnar et al., 2021b) and Cellular WL (Bodnar et al., 2021a):
Proposition C.12. Consider the SWL algorithm (Bodnar et al., 2021b), or more generally, the CWL
algorithms with either k-CL, k-IC, or k-C as lifting maps (k ≥ 3 is an integer) (Bodnar et al., 2021a,
Definition 14). These algorithms can neither distinguish whether a given graph has cut vertices nor
distinguish whether it has cut edges.
Proof. Observe that the counterexample graphs in both Examples C.9 and C.10 do not have cliques.
Therefore, SWL (or CWL with k-CL) reduces to the classic 1-WL and thus fails to distinguish
them. Since the lengths of any cycles in these counterexample graphs are at least m (m is defined in
Examples C.9 and C.10), we have that CWL with k-IC or k-C also reduces to 1-WL when m > k.
Therefore, there exists graphs whose size is O(k) such that CWL can neither distinguish cut vertices
nor cut edges.
Finally, we point out that even if k is not a constant (i.e., can scale to the graph size), CWL with k-IC
still fails to distinguish whether a given graph has cut vertices. This is because for Example C.9 with
k ≥ 2 (e.g. Figure 2(b,c)), CWL with IC still outputs the same graph representation for both G1
and G2. This happens because all the 2-dimensional cells in these examples are cycles of an equal
length of m + 2 and one can easily check that they have the same CWL color.
We finally turn to the case of subgraph-based WL variants.
Proposition C.13. The Overlap Subgraph WL (Wijesinghe & Wang, 2022) using any subgraph
mapping ω can neither distinguish whether a given graph has cut vertices nor distinguish whether
it has cut edges.
Proof. An important limitation of OS-WL is that if a graph does not contain triangles, then any over-
lap subgraph Suv between two adjacent nodes u, v will only have one edge {u, v}. Consequently,
the subgraph mapping ω does not take effect can OS-WL reduces to the classic 1-WL. Therefore,
Example C.9 with m > 1 and Example C.10 with m > 3 still apply here since the graphs G1 and G2
do not contain triangles (see Figure 2(a,b,d)). Moreover, Example C.9 with m = 1 (see Figure 2(c))
is also a counterexample as discussed in Wijesinghe & Wang (2022, Figure 2(a)).
25
𝑣𝑣2𝑚𝑚−1𝑚𝑚+2𝑚𝑚+12𝑚𝑚𝑚𝑚−1𝑚𝑚+1𝑚𝑚+21122……𝑚𝑚−1𝑚𝑚𝑚𝑚……2𝑚𝑚−12𝑚𝑚𝑛𝑛𝑛𝑛𝑣𝑣𝑣𝑣𝑣𝑣122𝑚𝑚−12𝑚𝑚2𝑚𝑚−12𝑚𝑚2𝑚𝑚+112𝑚𝑚−1𝑚𝑚𝑚𝑚+1𝑚𝑚+2𝑚𝑚+2…𝑚𝑚−1𝑚𝑚𝑚𝑚+1……𝑛𝑛2𝑚𝑚+13𝑚𝑚−13𝑚𝑚3𝑚𝑚+1…4𝑚𝑚−14𝑚𝑚3𝑚𝑚−13𝑚𝑚3𝑚𝑚+14𝑚𝑚−14𝑚𝑚𝑛𝑛2𝑚𝑚−1𝑣𝑣𝑣𝑣𝑛𝑛1𝑚𝑚−1𝑚𝑚𝑚𝑚+12𝑚𝑚𝑚𝑚+2…1𝑚𝑚−1𝑚𝑚𝑚𝑚+1𝑚𝑚+22𝑚𝑚−12𝑚𝑚𝑛𝑛22………2𝑣𝑣𝑣𝑣12𝑚𝑚−1𝑚𝑚𝑚𝑚+12𝑚𝑚−12𝑚𝑚𝑛𝑛2𝑚𝑚+13𝑚𝑚−13𝑚𝑚3𝑚𝑚+14𝑚𝑚−14𝑚𝑚1𝑚𝑚−1𝑚𝑚𝑚𝑚+1𝑚𝑚+22𝑚𝑚−12𝑚𝑚2𝑚𝑚+14𝑚𝑚−14𝑚𝑚𝑛𝑛𝑚𝑚+2…………3𝑚𝑚−13𝑚𝑚3𝑚𝑚+1Published as a conference paper at ICLR 2023
Proposition C.14. The DSS-WL with ego network policy without marking cannot distinguish the
graphs in Example C.9 with m = 1 (Figure 2(c)).
Proof. First note that for any two vertices u, v in either G1 or G2 defined in Example C.9, their
shortest path distance does not exceed 2. Thus we only need to consider the ego network policy
πEGO(1) and πEGO(2).
• For πEGO(2), the ego graphs of all nodes are simply the original graph and thus all graphs
in the bag Bπ and equal. Thus DSS-WL reduces to the classic 1-WL and cannot distinguish
G1 and G2.
• For πEGO(1), the ego graph of each node v ̸= n is a graph with 5 edges, having a shape of
two triangles sharing one edge. These ego graphs are clearly isomorphic. The ego graph of
the special node n is the original graph containing all edges. It is easy to see that the vertex
partition of DSS-WL becomes stable only after one iteration, and the color mapping of G1
and G2 is the same. Therefore, DSS-WL cannot distinguish G1 and G2.
We thus conclude the proof.
Proposition C.15. The GNN-AK architecture proposed in Zhao et al. (2022) cannot distinguish
whether a given graph has cut vertices.
Proof. The GNN-AK architecture is quite similar to DSS-WL using the ego network policy but is
weaker. There is also a subtle difference: GNN-AK adds the so-called centroid encoding. However,
unlike node marking that is performed before the WL procedure, centroid encoding is performed
after the WL procedure. The subtle difference causes GNN-AK to be unable to distinguish between
the two graphs G1 and G2.
We finally consider the DS-WL algorithm proposed in Cotta et al. (2021); Bevilacqua et al. (2022).
As discussed in Appendix B.4, the original DS-WL formulation only outputs a graph representation
rather than node colors. There are two simple ways to define nodes colors for DS-WL:
• If the graph generation policy π is node-based, then each subgraph in Bπ
i=1 is
uniquely associated to a specific node v ∈ V. We can thus use the graph representation of
each subgraph Gi as the color of each node. This strategy has appeared in prior works, e.g.
Zhao et al. (2022).
G = {{Gi}}|V|
• For a general graph generation policy π, there no longer exists an explicit bijective mapping
between nodes and subgraphs. In this case, another possible way is to define χG(v) :=
{{χGi(v) : Gi ∈ Bπ
G}}, similar to DSS-WL. This approach is recently introduced by Qian
et al. (2022). However, such a strategy loses the memory advantage of DS-WL (i.e., needing
Θ(|V||Bπ
G|)), and is less expressive than DSS-
WL. We thus do not study this variant in the present work.
G|) memory complexity rather than Θ(|V|+|Bπ
Proposition C.16. The DS-WL algorithm with node marking/deletion policy cannot distinguish cut
vertices when each node’s color is defined as its associated subgraph representation.
Proof. One can similarly check that for Example C.9 with m = 1 (see Figure 2(c)), the color of node
n will be the same for both graphs G1 and G2. Therefore, DS-WL cannot identify cut vertices.
Finally, using a similar proof technique, the NGNN architecture proposed in Zhang & Li (2021)
(with shortest path distance encoding) cannot identify cut vertices.
C.3 PROOF OF THEOREM 3.2
Theorem C.17. Let G = (V, EG) and H = (V, EH ) be two graphs, and let χG and χH be the
corresponding DSS-WL color mapping with node marking policy. Then the following holds:
• For any two nodes w ∈ V in G and x ∈ V in H, if χG(w) = χH (x), then w is a cut vertex
in graph G if and only if x is a cut vertex in graph H.
• For any two edges {w1, w2} ∈ EG and {x1, x2} ∈ EH , if {{χG(w1), χG(w2)}} =
{{χH (x1), χH (x2)}}, then {w1, w2} is a cut edge if and only if {x1, x2} is a cut edge.
26
Published as a conference paper at ICLR 2023
Proof. We divide the proof into two parts in Appendices C.3.1 and C.3.2, separately focusing on
proving each bullet of Theorem 3.2. Before going into the proof, we first define several notations.
Denote χu
G(v) as the color of node v under the DSS-WL algorithm when marking u as a special
node. By definition of DSS-WL (Line 7 in Algorithm 3), χG(v) = hash ({{χu
G(v) : u ∈ V}}). We
can similarly define the inverse mappings (χu
G)−1.
We first present a lemma which can help us exclude the case of disconnected graphs.
Lemma C.18. Given a node w, let SG(w) ⊂ V be the connected component in graph G that
comprises node w. For any two nodes w ∈ V in G and x ∈ V in H, if χG(w) = χH (x), then
χG[SG(w)](w) = χH[SH (x)](x).
Proof. We first prove that if χG(w) = χH (x), then {{χu
H (x) : u ∈
SH (x)}}. First note that for any nodes u, w in G and v, x in H, if u ∈ SG(w) but v /∈ SH (x),
then χu
H (x). This is because DSS-WL only performs neighborhood aggregation, and the
marking v cannot propagate to node x while the marking u can propagate to node w. By definition
we have
G(w) : u ∈ SG(w)}} = {{χu
G(w) ̸= χv
χG(w) = hash ({{χu
G(w) : u ∈ SG(w)}} ∪ {{χv
G(w) : v /∈ SG(w)}}) .
Similarly,
χH (x) = hash ({{χu
H (x) : u ∈ SH (x)}} ∪ {{χv
H (x) : v /∈ SH (x)}}) .
Since χG(w) = χH (x), we have {{χu
implies {{χu
χH[SH (x)](x).
G[SG(w)] : u ∈ SG(w)}} = {{χu
G : u ∈ SG(w)}} = {{χu
H : u ∈ SH (x)}}. This clearly
H[SH (x)] : u ∈ SH (x)}}, and thus χG[SG(w)](w) =
Note that w is a cut vertex in G implies w is a cut vertex in G[SG(w)]. Therefore, based on
Lemma C.18, we can restrict our attention to subgraphs G[SG(w)] and H[SH (x)] instead of the
original (potentially disconnected) graphs. In other words, in the subsequent proof we can simply
assume that both graphs G and H are connected.
We next present several simple but important properties regrading the DSS-WL color mapping as
well as the subgraph color mappings.
Lemma C.19. Let u, w be two nodes in connected graph G and v, x be two nodes in connected
graph H. Then the following holds:
(a) If w = u and x ̸= v, then χu
G(w) ̸= χv
H (x);
H (x), then χG(w) = χH (x);
H (x), then χG(u) = χH (v);
G(w) = χx
(b) If χu
G(w) = χv
G(w) = χv
(c) If χu
(d) χG(w) = χH (x) if and only if χw
(e) If χu
G(w) = χv
H (x), then disG(u, w) = disH (v, x).
H (x);
Proof. Item (a) holds because in DSS-WL, the node with marking cannot have the same color as
a node without marking. This can be rigorously proved by induction over the iteration t in the
DSS-WL algorithm (Line 6 in Algorithm 3).
Item (b) simply follows by definition of the DSS-WL aggregation procedure since the color χu
encodes the color of χG(w).
G(w)
We next prove item (c), which follows by using the WL-condition of DSS-WL algorithm (Proposi-
tion C.3). Since G is connected, there is a path from w to u. Therefore, in graph H there is also a
path from x to some node v′ satisfying χu
H (v′) (Lemma C.7). Now using item (a), it can
only be the case v′ = v and thus χu
G(u) = χv
H (v). Finally, by item (b) we obtain the desired result.
We next prove item (d). On the one hand, item (b) already shows that χw
G(x) =⇒
χG(w) = χH (x). On the other hand, by definition of the DSS-WL algorithm,
G(w) = χx
G(u) = χv
χG(w) = hash ({{χw
χH (x) = hash ({{χx
G(w)}} ∪ {{χu
H (x)}} ∪ {{χv
G(w) : u ∈ V\{w}}}) ,
H (x) : v ∈ V\{x}}}) .
27
Published as a conference paper at ICLR 2023
G(x).
G(w) ̸= χv
H (x) holds for all v ∈ V\{x} (by item (a)), we obtain
Since χG(w) = χH (x) and χw
G(w) = χx
χw
We finally prove item (e), which again can be derived from the WL-condition of DSS-WL al-
G(u))) =
gorithm.
disH (x, (χv
̸= v,
H (v′) ̸= χv
χv
H (v) =
χu
G(u). This yields disG(u, w) = disG(v, x) and concludes the proof.
G(u))). Using item (a), we have (χu
H (v). Therefore, it can only be the case that (χv
G(u)) = {u} and for any v′
G(u)) = {v} and χv
H )−1(χu
H (x), then by Corollary C.8 we have disG(w, (χu
If χu
H )−1(χu
G(w) = χv
G)−1(χu
G)−1(χu
C.3.1 PROOF FOR THE FIRST PART OF THEOREM 3.2
The following technical lemma is useful in the subsequent proof:
Lemma C.20. Let u, v ∈ V be two nodes in connected graphs G and H, respectively. If χu
H (v), then {{χu
χv
H (w) : w ∈ V}}.
G(w) : w ∈ V}} = {{χv
G(u) =
G := {{χu
H (w) : w ∈ N d
G(w) : w ∈ N d
G(u) := {w ∈ V : disG(u, w) = d} be the d-hop neighbors of node u in graph
G(u)}} as the multiset containing the color of all nodes
H (v) := {w : disH (v, w) = d} and
G = Cd
H .
Proof. Let N d
G, and denote Cd
w with distance d to node u. We can similarly denote N d
H (v)}}. It suffices to prove that for all d ∈ N+, Cd
H = {{χv
Cd
We will prove the above result by induction. The case of d = 0 is trivial. Now suppose the case
of d is true (i.e., Cd
H . Note that for any nodes x1, x2
satisfying χu
H (w) : w ∈ NH (x2)}}. Therefore,
by the induction assumption Cd
G = Cd+1
G(w) : w ∈ NG(x1)}} = {{χv
H ) and we want to prove Cd+1
G(x1) = χv
H (x2), {{χu
G = Cd
G = Cd
H ,
(cid:91)
{{χu
G(w) : w ∈ NG(x)}} =
(cid:91)
{{χv
H (w) : w ∈ NH (x)}}.
x∈N d
G(u)
x∈N d
H (v)
We next claim that Cd
the same color χu
this property, we obtain
G ∩ Cd′
G(w1) = χu
G = ∅ for any d ̸= d′. This is because for any nodes w1 and w2 with
G(w2), by Lemma C.19(e) we have disG(w1, u) = disG(w2, u). Using
(cid:91)
{{χu
G(w) : w ∈ NG(x) ∩ N d+1
G (u)}} =
x∈N d
G(u)
It is equivalent to the following equation:
(cid:91)
{{χu
G(w)}} × |NG(w) ∩ N d
G(u)| =
(cid:91)
x∈N d
H (v)
(cid:91)
{{χv
H (w) : w ∈ NH (x) ∩ N d+1
H (v)}}.
{{χv
H (w)}} × |NH (w) ∩ N d
H (v)|.
w∈N d+1
G (u)
w∈N d+1
H (v)
where {{c}} × m is a multiset containing m repeated elements c. Finally, observe that if χu
H (w2) for some nodes w1 and w2, then |NG(w1) ∩ N d
χv
G ∩ Cd′
Cd
G = ∅ for any d ̸= d′). Consequently, {{χu
H (v)}}, namely Cd+1
N d+1
H . We have thus completed the proof of the induction step.
G(u)| = |NH (w2) ∩ N d
G(w) : w ∈ N d+1
G (u)}} = {{χv
G = Cd+1
G(w1) =
H (v)| (because
H (w) : w ∈
We now present the following key result, which shows an important property of the color mapping
for DSS-WL:
Corollary C.21. Let u, v ∈ V be two nodes in connected graph G with the same DSS-WL color, i.e.
χG(u) = χG(v). Then for any color c ∈ C, {{χu
G(w) : w ∈ χ−1
G(w) : w ∈ χ−1
G (c)}} = {{χv
G (c)}}.
Proof. First observe that if χG(u) = χG(v), then χu
G(w) : w ∈ V}} = {{χv
sequently, {{χu
w ∈ χ−1
G(w) : w ∈ χ−1
and w2 /∈ χ−1
χG(w1) = χG(w2), yielding a contradiction.
G (c), such that χu
G(w) : w ∈ V}} holds by Lemma C.20.
G (c)}}, then there must exist two nodes w1 ∈ χ−1
G(v) (by Lemma C.19(d)). Con-
G(w) :
G (c)
G(w2). Therefore, by Lemma C.19(b) we have
G (c)}} ̸= {{χv
G(w1) = χv
G(u) = χv
If {{χu
In the subsequent proof, we assume the connected graph G is not vertex-biconnected and let u ∈ V
be a cut vertex in G. Let {Si}m
i=1 (m ≥ 2) be the partition of the vertex set V\{u}, representing
each connected component after removing node u.
28
Published as a conference paper at ICLR 2023
(a) Proof of Lemma C.22
(b) Proof of Lemma C.23
Figure 4: Several illustrations to help understand the lemmas.
Lemma C.22. There is at most one set Si satisfying Si ∩ χ−1
Si ∩ χ−1
G (χG(u)) ̸= ∅ for some i ∈ [m], then for any j ∈ [m] and j ̸= i, Sj ∩ χ−1
G (χG(u)) ̸= ∅. In other words, if
G (χG(u)) = ∅.
G (χG(u))| = 1, the conclusion clearly holds. If |χ−1
Proof. When |χ−1
G (χG(u))| > 1, then we can
pick a node u1 ∈ χ−1
G (χG(u)) that maximizes the shortest path distance disG(u1, u). Let u1 ∈ Si
for some i ∈ [m]. If the lemma does not hold, then we can pick another node u2 ∈ χ−1
G (χG(u)) and
u2 /∈ Si. Since u1 and u2 are in different connected component after removing u, disG(u1, u2) =
disG(u1, u) + disG(u2, u). See Figure 4(a) for an illustration of this paragraph.
By Corollary C.21, {{χu1
G(w) : w ∈ χ−1
G (w) : w ∈ χ−1
G (χG(u))}}. There-
fore, there must exist a node u3 ∈ χ−1
G (u2) = χu
G(u3). We thus have
disG(u2, u1) = disG(u3, u) by Lemma C.19(e). On the other hand, by definition of the node u1,
disG(u1, u) ≥ disG(u3, u). Therefore, disG(u2, u1) = disG(u1, u) + disG(u2, u) > disG(u3, u).
This yields a contradiction and concludes the proof.
G (χG(u))}} = {{χu
G (χG(u)) satisfying χu1
Lemma C.23. For all u′ ∈ χ−1
G (χG(u)), u′ it is a cut vertex of G.
G(u) = χu′
Proof. When |χ−1
G (χG(u))| = 1, the conclusion clearly holds. Now assume |χ−1
G (χG(u))| > 1.
Since u is a cut vertex in G, by Lemma C.22, there exists a set Sj such that Sj ∩ χ−1
G (χG(u)) = ∅.
Pick any node w ∈ Sj, then χG(w) ̸= χG(u). Let u′ ̸= u be any node with color χG(u) = χG(u′).
G (u′) by Lemma C.19(d). Based on the WL-condition of the mappings
It follows that χu
G , by Lemma C.7 there exists a node w′ with color χu′
G and χu′
χu
G (w′) = χu
G(w) (because there is a
path from node u to w). See Figure 4(b) for an illustration of this paragraph.
Suppose u′ is not a cut vertex. Then there is a path P from w′ to u without going through node u′.
Denote P = (x0, · · · , xd) where x0 = w′ and xd = u. It follows that χu′
G (u′) for all
i ∈ [d] (by Lemma C.19(a)). Again by using the WL-condition, there exists a path Q = (y0, · · · , yd)
satisfying y0 = w and χu
G (u), which
implies χG(yd) = χG(u) by using Lemma C.19(b). By the definition of w and Lemma C.22, any
path from w to yd ∈ χ−1
G(u) for
some i ∈ [d]. However, we have proved that χu
G(u), yielding a
contradiction. Therefore, u′ is a cut vertex.
G (χG(u)) must go through node u, implying that χu
G (xi) for all i ∈ [d]. In particular, χu
G (xi) ̸= χu′
G (xi) ̸= χu′
G(yd) = χu′
G(yi) = χu′
G(yi) = χu′
G (u′) = χu
G(yi) = χu
G (χG(u))| = |χ−1
Using a similar proof technique as the one in Lemma C.23, we can prove the first part of Theo-
rem 3.2. Suppose u′ ∈ χ−1
H (χG(u)) and we want to prove that u′ is a cut vertex of graph H.
Observe that |χ−1
H (χH (u))|. (A simple proof is as follows: χG(u) = χH (u′)
G(u) = χu′
H (u′) by Lemma C.19(d), and thus using Lemma C.20 we have {{χu
implies χu
G(w) : w ∈
V}} = {{χu′
H (w) : w ∈ V}} and finally obtain {{χG(w) : w ∈ V}} = {{χH (w) : w ∈ V}} by
Lemma C.19(b).)
We first consider the case when |χ−1
we can similarly pick w ∈ Sj in G and w′ in H satisfying χG(w) ̸= χG(u) and χu′
Since |χ−1
H (χH (u))| > 1. Following the above proof,
G(w).
H (χG(u)) in H such that uH ̸= u′. If u′ is
G (χG(u))| > 1, we can pick a node uH ∈ χ−1
G (χG(u))| = |χ−1
H (w′) = χu
29
𝑢𝑢𝑆𝑆𝑖𝑖𝑢𝑢1𝑢𝑢2𝑢𝑢3𝑢𝑢𝑆𝑆𝑖𝑖𝑢𝑢′𝑤𝑤𝑤𝑤′𝑆𝑆𝑗𝑗𝑦𝑦𝑑𝑑𝑃𝑃𝑄𝑄Published as a conference paper at ICLR 2023
(a) Proof of the main theorem (|χ−1
G (χG(u))| > 1)
(b) Proof of the main theorem (|χ−1
G (χG(u))| = 1)
Figure 5: Several illustrations to help understand the main proof of Theorem 3.2.
G(yi) = χu
G(yi) = χu′
H (xi) ̸= χu′
H (xi) ̸= χu′
G (χG(u)) must go through node u, implying that χu
not a cut vertex, then there is a path P = (x0, · · · , xd) in H where x0 = w′ and xd = uH , such
that χu′
H (u′) for all i ∈ [d] (by Lemma C.19(a)). Using the WL-condition, there exists a
G(yi) = χu′
path Q = (y0, · · · , yd) in G satisfying y0 = w and χu
H (xi) for all i ∈ [d]. In particular,
G(yd) = χu′
χu
H (uH ), which implies χG(yd) = χG(uH ) by using Lemma C.19(b). However, any
path from w to yd ∈ χ−1
G(u) for some
i ∈ [d]. This yields a contradiction because χu
G(u). See Figure 5(a)
for an illustration of this paragraph.
We finally consider the case when |χ−1
G (χG(u))| = |χ−1
H (χH (u))| = 1. Let w ∈ S1 and x ∈ S2
be two nodes in G that belongs to different connected components when removing node u, then
χG(w) ̸= χG(u) and χG(x) ̸= χG(u). Since χG(u) = χH (u′), by the WL-condition (Lemma C.7)
G(w) = χw′
there is a node w′ ∈ χ−1
H (w′) (Lemma C.19(d)). Again
by the WL-condition, there is a node x′ ∈ (χw′
G(x)) in H. Clearly, w′ ̸= u′ and x′ ̸= u′
(because they have different colors). If u′ is not a cut vertex, then there is path P = (y0, · · · , yd) in
H such that y0 = x′, yd = w′ and yi ̸= u′ for all i ∈ [d]. It follows that for all i ∈ [d], χH (yi) ̸=
χH (u′) by our assumption |χ−1
H (u′) (by Lemma C.19(b)).
H (x′), by the WL-condition (Lemma C.7), there is a path Q = (z0, · · · , zd) in G
Since χw
satisfying z0 = x and zi ∈ (χw
H (yi)) for i ∈ [d]. See Figure 5(b) for an illustration of this
paragraph.
H (χG(w)) in H. Consequently, χw
H )−1(χw
H (χH (u))| = 1, and thus χw′
H (yi) ̸= χw′
G(x) = χw′
H (u′) = χu
G)−1(χw′
H (w′) and Lemma C.19(a). On the other hand, by
Clearly, we have zd = w using χw
Lemma C.19(b), χw
H (yi) implies χG(zi) = χH (yi) and thus χG(zi) ̸= χH (u′) = χG(u)
holds for all i ∈ [d] and thus zi ̸= u. In other words, we have found a path from x to w without
going through node u, which yields a contradiction as u is a cut vertex. We have thus finished the
proof.
G(zi) = χw′
G(zd) = χw′
C.3.2 PROOF FOR THE SECOND PART OF THEOREM 3.2
The proof is based on the following key result:
Corollary C.24. Let w and x be two nodes in connected graph G with the same DSS-WL color, i.e.
χG(w) = χG(x). Then for any color c ∈ C,
{{disG(w, v) : v ∈ χ−1
G (c)}} = {{disG(x, v) : v ∈ χ−1
G (c)}}.
Proof. By Corollary C.21, we have {{χw
any nodes u, v, χw
obtained the desired conclusion.
G(u) = χx
G (c)}}. Since for
G(v) implies disG(u, w) = disG(v, x) (by Lemma C.19(e)), we have
G (c)}} = {{χx
G(v) : v ∈ χ−1
G(v) : v ∈ χ−1
Equivalently, the above corollary says that if χG(w) = χG(x), then the following two multisets are
equivalent:
{{(disG(w, v), χG(v)) : v ∈ V}} = {{(disG(x, v), χG(v)) : v ∈ V}}.
Therefore, it guarantees that the vertex partition induced by the DSS-WL color mapping is finer
than that of the SPD-WL (Algorithm 4 with dG = disG). We can thus invoke Theorem 4.1, which
directly concludes the proof (due to Proposition C.56).
30
𝑢𝑢𝑆𝑆𝑖𝑖𝑤𝑤𝑆𝑆𝑗𝑗𝑦𝑦𝑑𝑑𝑄𝑄𝑢𝑢′𝑤𝑤′𝑢𝑢𝐻𝐻𝑃𝑃Graph𝐺𝐺Graph𝐻𝐻𝑢𝑢𝑆𝑆𝑖𝑖𝑤𝑤𝑆𝑆𝑗𝑗𝑥𝑥𝑄𝑄𝑢𝑢′𝑤𝑤′𝑥𝑥′𝑃𝑃Graph𝐺𝐺Graph𝐻𝐻Published as a conference paper at ICLR 2023
C.4 PROOF OF THEOREM 4.1
Theorem C.25. Let G = (V, EG) and H = (V, EH ) be two graphs, and let χG and χH be the
corresponding SPD-WL color mapping. Then the following holds:
• For any two edges {w1, w2} ∈ EG and {x1, x2} ∈ EH , if {{χG(w1), χG(w2)}} =
{{χH (x1), χH (x2)}}, then {w1, w2} is a cut edge if and only if {x1, x2} is a cut edge.
• If the graph representations of G and H are the same under SPD-WL, then their block
cut-edge trees (Definition 2.3) are isomorphic. Mathematically, {{χG(w) : w ∈ V}} =
{{χH (w) : w ∈ V}} implies that BCETree(G) ≃ BCETree(H).
Proof Sketch. The proof of Theorem 4.1 is highly non-trivial and is divided into three parts (pre-
sented in Appendices C.4.1 to C.4.3, respectively). We first consider the special setting when both
G and H are connected and {{χG(w) : w ∈ V}} = {{χH (w) : w ∈ V}}. Assume G is not
edge-biconnected, and let {u, v} ∈ EG be a cut edge in G. We separately consider two cases:
χG(u) ̸= χG(v) (Appendix C.4.1) and χG(u) = χG(v) (Appendix C.4.2), and prove that any edge
{u′, v′} ∈ EH satisfying {{χG(u), χG(v)}} = {{χH (u′), χH (v′)}} is also a cut edge of H. This
basically finishes the proof of the first bullet in the theorem. Finally, we consider the general setting
where graphs G, H can be disconnected and their representation is not the same in Appendix C.4.3,
and complete the proof of Theorem 4.1.
Without abuse of notation, throughout Appendices C.4.1 and C.4.2 we redefine the color set C :=
{χG(w) : w ∈ V} = {χH (w) : w ∈ V} to focus only on colors that are present in G (or H), rather
than all (irrelevant) colors in the range of a hash function.
C.4.1 THE CASE OF χG(u) ̸= χG(v) FOR CONNECTED GRAPHS
We first define several notations. Throughout this case, denote {Su, Sv} as the partition of V, rep-
resenting the two connected components after removing the edge {u, v} such that u ∈ Su, v ∈ Sv,
Su ∩ Sv = ∅ and Su ∪ Sv = V. We then define an important concept called the color graph.
Definition C.26. (Color graph) Define the auxiliary color graph GC = (C, EGC ) where EGC =
{{{χG(w), χG(x)}} : {w, x} ∈ EG}. Note that GC can have self loops, so each edge is denoted as
a multiset with two elements.
Lemma C.27. Let S = χ−1
or χG(v). Then either S ∩ Su = {u} or S ∩ Sv = {v}.
G (χG(v)) be the set containing vertices with color χG(u)
G (χG(u)) ∪ χ−1
Proof. Assume the lemma does not hold, i.e. |S ∩ Su| > 1 and |S ∩ Sv| > 1. We first prove that
G (χG(u)) ∩ Sv ̸= ∅ and χ−1
χ−1
G (χG(v)) ∩ Su ̸= ∅. By symmetry, we only need to prove the former.
Suppose χ−1
G (χG(v)) ∩ Sv)\{v} ̸= ∅ (because |S ∩ Sv| > 1), and
thus there exists v′ ∈ Sv, v′ ̸= v such that χG(v′) = χG(v). Note that v′ must connect to a node u′
with χG(u′) = χG(u). Since {u, v} is a cut edge in G, u′ ∈ Sv. Therefore, χ−1
G (χG(u)) ∩ Sv ̸= ∅,
yielding a contradiction. This paragraph is illustrated in Figure 6(a).
G (χG(u)) ∩ Sv = ∅, then (χ−1
G (χG(u)) ∩ Su)\{u} ̸= ∅; (ii) (χ−1
We next prove that at least one of the following two conditions holds (which are symmetric): (i)
(χ−1
G (χG(v)) ∩ Sv)\{v} ̸= ∅. Based on the above paragraph,
there exists v′ ∈ Su satisfying χG(v′) = χG(v). Note that v′ must connect to a node with color
χG(u). If condition (i) does not hold, i.e. χ−1
G (χG(u)) ∩ Su = {u}, then v′ must connect to u. This
means |NG(u) ∩ χ−1
G (χG(u)) ∩ Sv ̸= ∅ (the above paragraph), we
can pick such a node u′ ∈ χ−1
G (χG(u)) ∩ Sv. By the WL-condition (Proposition C.4), |NG(u′) ∩
G (χG(v))| ≥ 2, which implies |Sv ∩ χ−1
χ−1
G (χG(v)) ∩ Sv)\{v} ̸= ∅
holds, which is exactly the condition (ii). This paragraph is illustrated in Figure 6(b).
G (χG(v))| ≥ 2. Again using χ−1
G (χG(v))| ≥ 2. Thus (χ−1
Based on the above two paragraphs, by symmetry we can without loss of generality assume
G (χG(u)) ∩ Sv ̸= ∅ and (χ−1
χ−1
G (χG(u)) ∩ Su)\{u} ̸= ∅. We are now ready to derive a contradic-
tion. To do this, pick ˜u = arg maxw∈χ−1
G (χG(u)) disG(u, w) and separately consider the following
two cases:
• ˜u ∈ Su. Then by picking a node x ∈ Sv ∩ χ−1
G (χG(u)), it follows that disG(x, ˜u) =
disG(x, v) + disG(u, ˜u) + 1 > disG(u, ˜u).
31
Published as a conference paper at ICLR 2023
(a)
(b)
Figure 6: Illustration of the proof of Lemma C.27.
Figure 7:
Lemma C.28.
Illustration of the proof of
• ˜u ∈ Sv. Then by picking a node x ∈ (Su ∩ χ−1
G (χG(u)))\{u}, it follows that disG(x, ˜u) ≥
disG(x, u) + disG(u, ˜u) > disG(u, ˜u).
In both cases, x and u cannot have the same color under SPD-WL because
max
G (χG(u))
w∈χ−1
disG(u, w) = disG(u, ˜u) < disG(x, ˜u) ≤
max
G (χG(u))
w∈χ−1
disG(x, w).
This yields a contradiction and concludes the proof.
Based on Lemma C.27,
χ−1
G (χG(u)) ∩ Su = {u} and χ−1
Lemma C.28. For any u1, u2 ∈ χ−1
node v′ ∈ χ−1
G (χG(v)).
in the subsequent proof we can without loss of generality assume
G (χG(v)) ∩ Su = ∅. This leads to the following lemma:
G (χG(u)), u1 ̸= u2, any path from u1 to u2 goes through a
1 to u′
G (χG(u)) ∩ Su = {u}. If |χ−1
2 without going through any node in the set χ−1
Proof. Note that χ−1
G (χG(u)) ∩ Sv| ≤ 1, the conclusion is clear
since any path from u1 to u2 goes through v. Now suppose |χ−1
G (χG(u)) ∩ Sv| > 1 and the
2 ∈ χ−1
1, u′
lemma does not hold. Then there exist two different nodes u′
G (χG(u)) ∩ Sv and a
path P from u′
G (χG(v)). Pick u1, u2 and
P such that the length |P | is minimal. Split P into two parts P1 and P2 with endpoints {u1, w}
and {w, u2} such that |P1| ≤ |P2| ≤ |P1| + 1 and |P1| + |P2| = |P |. Note that |P | ≥ 2 since
{u1, u2} /∈ EG (otherwise u cannot have the same color as u1 because χ−1
G (u) ∩ Su = {u}).
Therefore, w ̸= u1 and w ̸= u2. Also note that χG(w) ̸= χG(u) since |P | is minimal. Since
SPD-WL satisfies the WL-condition (Proposition C.4), there is a path (not necessarily simple) from
u to some w′ ∈ χ−1
G (χG(v))
(according to Lemma C.7). Therefore, w′ ∈ Su. See Figure 6 for an illustration of this paragraph.
We next prove that disG(u, w′) = |P1|. First, we obviously have disG(u, w′) ≤ |P1|. Moreover,
since w′, u ∈ Su and χ−1
G (χG(v)) ∩ Su = ∅ (Lemma C.27), any shortest path from w′ to u does not
go through nodes in the set χ−1
G (χG(v)). Again using the WL-condition, there exists a path P3 (not
necessarily simple) from w to some u3 ∈ χ−1
G (χG(u)) of length |P3| = disG(u, w′) without going
through nodes in the set χ−1
G (χG(v)) (according to Lemma C.7). It follows that u3 ∈ Sv. Consider
the following two cases:
G (χG(w)) of length |P1| without going through nodes in the set χ−1
• If u3 = u1, by the minimal length of P we have |P1| ≤ |P3| = disG(u, w′) ≤ |P1| and thus
disG(u, w′) = |P1|.
• If u3 ̸= u1, by linking the path P1 and P3, there will be a path of length |P1| + |P3| from
u1 to u3 without going through nodes in χ−1
G (χG(v)). Since P has the minimal length,
|P1| + |P2| ≤ |P1| + |P3|. Therefore, |P2| ≤ |P3| = disG(u, w′) and thus by definition
|P1| ≤ |P2| ≤ disG(u, w′) ≤ |P1|. Therefore, |P1| = |P2| = disG(u, w′).
32
𝑢𝑢𝑆𝑆𝑢𝑢𝑣𝑣𝑆𝑆𝑣𝑣𝑢𝑢𝑢𝑣𝑣𝑢𝑢𝑢𝑆𝑆𝑢𝑢𝑣𝑣𝑆𝑆𝑣𝑣𝑢𝑢𝑢𝑣𝑣1𝑣𝑣2𝑣𝑣𝑢𝑢𝑢3𝑆𝑆𝑢𝑢𝑣𝑣𝑆𝑆𝑣𝑣𝑤𝑤𝑢𝑢1𝑢𝑢2𝑢𝑢𝑤𝑤𝑤𝑃𝑃1𝑃𝑃2𝑃𝑃3Published as a conference paper at ICLR 2023
Now define the set D(x) := {u′ : u′ ∈ χ−1
G (χG(u)), disG(x, u′) ≤ |P2|}. Let us focus on the
cardinality of the sets D(w) and D(w′). It follows that D(w′) = {u}, because for any other node
u′ ∈ χ−1
G (χG(u)), u′ ̸= u, we have u′ ∈ Sv and thus
disG(w′, u′) > disG(w′, v) = disG(w′, u) + 1 = |P1| + 1 ≥ |P2|.
Therefore, |D(w′)| = 1. On the other hand, we clearly have |D(w)| ≥ 2 since both u1, u2 ∈
D(w). Consequently, w and w′ cannot have the same color under the SPD-WL algorithm because
|D(w′)| ̸= |D(w′)|. This yields a contradiction and completes the proof.
The next lemma presents an important property of the color graph GC (defined in Definition C.26).
Lemma C.29. GC has a cut edge {{χG(u), χG(v)}}.
Proof. Suppose {{χG(u), χG(v)}} is not a cut edge of GC. Then there is a simple cycle (c1, · · · , cm)
where c1 = χG(u), cm = χG(v) and m > 2. Namely, there exists a simple path from c1 to cm with
length ≥ 2. By the definition of GC and the WL-condition, there exists a sequence of nodes of G
{wi}m
i=1 where w1 = u and χ(wi) = ci such that {wi, wi+1} ∈ EG, i ∈ [m − 1]. Note that wi ̸= u
for i = {2, · · · , m} and w2 ̸= v because (c1, · · · , cm) is a simple path. Therefore, wi ∈ Su for all
i ∈ [m]. However, it contradicts |S ∩ Su| = 1 (Lemma C.27) since χG(wm) = χG(v).
Combining Lemmas C.27 to C.29, we arrived at the following corollary:
Corollary C.30. For all u′ ∈ χ−1
edge of G.
G (χG(u)) and v′ ∈ χ−1
G (χG(v)), if {u′, v′} ∈ EG, then it is a cut
Proof. If {u′, v′} is not a cut edge, there is a simple cycle going through {u′, v′}. Denote it as
(w1, · · · , wm) where w1 = u′, wm = v′, m > 2. By Lemma C.27, w2 /∈ χG(v), otherwise u′ will
connect to at least two different nodes w2, wm ∈ χ−1
G (χG(v)) and thus u′ and u cannot have the
same color under SPD-WL. Let j be the index such that j = min{j ∈ [m] : χG(wj) = χG(v)}, then
j > 2. Consider the path (w1, · · · , wj). It follows that χG(wk) ̸= χG(u) for all k ∈ {2, · · · , j}
by Lemma C.28 (otherwise there is a path from node w1 to some node wi ∈ χ−1
G (χG(u)) (i ∈
{2, · · · , j}) that does not go through nodes in the set χ−1
G (χG(v)), a contradiction). Therefore,
(χG(w1), · · · , χG(wj)) is a path of length ≥ 2 in GC from χG(u) to χG(v) (not necessarily simple),
without going through the edge {{χG(u), χG(v)}}. This contradicts Lemma C.29, which says that
{{χG(u), χG(v)}} is a cut edge in GC.
Based on Lemma C.29, the cut edge {{χG(u), χG(v)}} partitions the vertices C of the color graph
GC into two classes. Denote them as {Cu, Cv} where χG(u) ∈ Cu and χG(v) ∈ Cv. The next
corollary characterizes the structure of the node colors calculated in SPD-WL.
Corollary C.31. For any w satisfying χG(w) ∈ Cu, there exists a cut edge {u′, v′}, u′ ∈
χ−1
G (χG(u)), v′ ∈ χ−1
G (χG(v)), that partitions V into two classes Su′ ∪ Sv′, u′, w ∈ Su′, v′ ∈ Sv′,
such that χ−1
Remark C.32. Corollary C.31 can be seen as a generalized version of Lemma C.27. Indeed, when
w ∈ Su, one can pick u′ = u and v′ = v. Then χ−1
G (χG(v′)) ∪
Su′ = ∅ hold due to Lemma C.27. In general, Corollary C.31 says that all the cut edges with color
{χG(u), χG(v)} play an equal role: Lemma C.27 applies for any chosen cut edge {u′, v′}. An
illustration of Corollary C.31 is given in Figure 8(a).
G (χG(u′)) ∪ Su′ = {u′} and χ−1
G (χG(u′)) ∪ Su′ = {u′} and χ−1
G (χG(v′)) ∪ Su′ = ∅.
Proof. By the definition of Cu, any node c ∈ Cu in the color graph can reach the node χG(u)
without going through χG(v). Therefore, there exists some u′ ∈ χ−1
G (χG(u)) such that there exists
a path P1 from w to u′ without going through nodes in the set χ−1
G (χG(v)). Also, there exists a
node v′ ∈ NG(u′) with χG(v′) = χG(v) due to the color of u′. By Corollary C.30, {u′, v′} is a cut
edge of G. Clearly, w ∈ Su′.
We next prove the following fact: for any x ∈ Su′, χG(x) ∈ Cu. Otherwise, one can pick a node
x ∈ Su′ with color χG(x) ∈ Cv. Consider the shortest path between nodes x and u′, denoted
as (y1, · · · , ym) where y1 = x and ym = u′. It follows that yi ∈ Su for all i ∈ [m]. Denote
33
Published as a conference paper at ICLR 2023
(a)
Figure 8: Illustration of Corollary C.31 and its proof.
(b)
ci = χG(yi), i ∈ [m]. Then (c1, · · · , cm) is a path (not necessarily simple) in the color graph GC.
Now pick the index j = max{j ∈ [m] : cj ∈ Cv} (which is well-defined because c1 ∈ Cv). It
follows that j < m (since ym ∈ Cu), cj = χG(v) and cj+1 = χG(u) (because {{χG(u), χG(v)}} is
a cut edge that partitions the color graph GC into Cu and Cv). Consider the following two cases (see
Figure 8(b) for an illustration):
• j = m−1. Then u′ connects to both nodes yj and v′ with color χG(yj) = χG(v′) = χG(v).
This contradicts Lemma C.27 since u only connects to one node v with color χG(v).
• j < m − 1. Then yj+1 ̸= u′ because the path (y1, · · · , ym) is simple. Howover, one has
χG(yi) ̸= χG(v) for all i ∈ {j +1, · · · , m} by definition of j. This contradicts Lemma C.28.
This completes the proof that for any x ∈ Su′, χG(x) ∈ Cu. Therefore, χ−1
G (χG(v′)) ∪ Su′ = ∅.
We finally prove that χ−1
G (χG(u)) ∪ Su′ and u′′ ̸= u′.
By Lemma C.28, the shortest path between u′ and u′′ goes through some node v′′ with color χG(v).
Clearly, v′′ ∈ Su, which contradicts the above paragraph and concludes the proof.
G (χG(u)) ∪ Su′ = {u′}. If not, pick u′′ ∈ χ−1
We have already fully characterized the properties of cut edges {u′, v′} with color {χG(u), χG(v)}.
Now we switch our focus to the graph H. We first prove a general result that holds for arbitrary H.
Lemma C.33. Let {w1, w2} ∈ EH and P is a path with the minimum length from w1 to w2 without
going through edge {w1, w2}. In other words, linking path P with the edge {w1, w2} forms a simple
cycle Q. Then for any two nodes x1, x2 in Q, disH (x1, x2) = disQ(x1, x2).
Proof. Split the cycle Q into two paths Q1 and Q2 with endpoints {x1, x2} where Q1 contains
the edge {w1, w2} and Q2 does not contain {w1, w2}. Assume the above lemma does not hold
and disH (w, x) < disQ(w, x). It means that there exists a path R in H from x1 to x2 for which
|R| < min(|Q1|, |Q2|). Note that the edge {u, v} occurs at most once in R. Separately consider
two cases:
• {w1, w2} occurs in R. Then linking R with Q2 forms a cycle that contains {w1, w2} exactly
once;
• {w1, w2} does not occur in R. Then linking R with Q1 forms a cycle that contains {w1, w2}
exactly once.
In both cases, the cycle has a length less than |Q|. This contradicts the condition that P is a path
with minimum length from w1 to w2 without passing edge {w1, w2}.
We can similarly consider the color graph H C = (C, EH C ) defined in Definition C.26. Note that we
have assumed that the graph representations of G and H are the same, i.e. {{χG(w) : w ∈ V}} =
{{χH (w) : w ∈ V}}. It follows that H C is isomorphic to GC and the identity vertex mapping is
an isomorphism, i.e., {{c1, c2}} ∈ EGC ⇐⇒ {{c1, c2}} ∈ EH C . Therefore, {{χG(u), χG(v)}} is a
cut edge of H C (Lemma C.29) that splits the vertices C into two classes Cu, Cv. Since the vertex
labels of H are not important, we can without abuse of notation let u, v be two nodes such that
34
𝑣𝑣1𝑣𝑣2𝑣𝑣3𝑣𝑣4𝑢𝑢1𝑢𝑢2𝑢𝑢3𝑢𝑢4𝑢𝑢5𝑢𝑢6𝑢𝑢7𝑢𝑢8𝑤𝑤2𝑤𝑤5𝑤𝑤7𝑤𝑤3𝑤𝑤6𝑤𝑤4𝑤𝑤1𝑤𝑤8𝑦𝑦𝑗𝑗𝑆𝑆𝑢𝑢𝑢𝑢′𝑆𝑆𝑣𝑣𝑤𝑤𝑥𝑥𝑣𝑣′𝑃𝑃1𝑦𝑦𝑗𝑗+1Published as a conference paper at ICLR 2023
(a) Graph H
(b) Graph G
Figure 9: Illustrations to help understand the proof of the main result.
{u, v} ∈ EH , χH (u) = χG(u), χH (v) = χG(v), and χH (u) ∈ Cu, χH (v) ∈ Cv. We similarly
define χ−1
H (c) = {w ∈ V : χH (w) = c}. Define a mapping h : C → {χH (u), χH (v)} where
h(c) =
(cid:26) χH (u)
χH (v)
if disH C (c, χH (u)) < disH C (c, χH (v)),
if disH C (c, χH (u)) > disH C (c, χH (v)).
(11)
Note that it never happens that disH C (c, χH (u)) = disH C (c, χH (v)) because {{χH (u), χH (v)}} is
a cut edge of H C.
Assume {u, v} is not a cut edge in H. Then there exists a path (w1, · · · , wm) in H with w1 =
u and wm = v without going through {u, v}. We pick such a path with the minimum length,
then the path is simple. Since h(χH (u)) ∈ Cu and h(χH (v)) ∈ Cv, there is a minimum index
j ∈ [m − 1] such that h(χH (wj)) ∈ Cu and h(χH (wj+1)) ∈ Cv. By definition of Cu, Cv and the
cut edge {{χH (u), χH (v)}}, it follows that χH (wj) = χH (u) and χH (wj+1) = χH (v). Denote
u′ := wj. Note that j ̸= 1 and j ̸= 2, otherwise u either connects to two nodes w2 and wm
with color χH (w2) = χH (wm) = χH (v), or connects to the node u′ with color χH (u′) = χH (u),
contradicting χH (u) = χG(u). Pick k = ⌈j/2⌉. By Lemma C.33, (w1, · · · , wk) is the shortest path
between u and wk, and (wk, · · · , wj) is the shortest path between wk and u′. We give an illustration
of the structure of H in Figure 9(a) based on this paragraph.
Since the graph representations of G and H are the same under SPD-WL, there exists a node w′
with color χG(w′) = χH (wk) and two different nodes u′
2 with color χG(u′
1, u′
2) =
χG(u), such that disG(w′, u′
1) = disH (wk, u1) and disG(w′, u′
2) = disH (wk, u2). In particular,
|disG(w′, u′
1) − disG(w′, u′
2)| ≤ 1. Note that by the definition of indices j and k, in the color graph
H C there is a path from χH (wk) to χH (u) without going through nodes in the set χ−1
H (χH (v)),
w, v′
so χH (wk) ∈ Cu, namely χG(w′) ∈ Cu. By Corollary C.31, there is a cut edge {u′
w} that
w ̸=
partitions G into two vertex sets Su′
1 and u′
u′
w with color
χG(u′) = χG(u) must first go through u′
1) −
disG(w′, u′
w) and
disG(w′, u′
w). We give an illustration of the structure of G in Figure 9(b) based on
this paragraph.
Pick any vw ∈ χ−1
w, w′). Denote the operation
dropmin(S) := S\{{min S}} that takes a multiset S and removes one of the minimum elements
in S. We have
w , with w′, u′
2 (otherwise by Corollary C.31 any path from w′ to a node u′
2)| ≥ 2 and yielding a contradiction). Therefore, disG(w′, u′
2) > disG(w′, u′
H (χH (v)) satisfying disH (vw, wk) = disG(v′
w, implying that |disG(w′, u′
w and then go through v′
1) > disG(w′, u′
w . Note that u′
1) = χG(u′
w ∈ Su′
w ∈ Sv′
w ̸= u′
w , v′
̸= u′
, Sv′
w
dropmin({{disG(w′, uG) : uG ∈ χ−1
G (χG(u)))
w, uG) : uG ∈ χ−1
= dropmin({{disG(w′, v′
= dropmin({{disH (wk, vw) + disH (vw, uH ) : uH ∈ χ−1
w) + disG(v′
G (χG(u))}})
H (χH (u))}})
(by Corollary C.31)
and also
dropmin({{disG(w′, uG) : uG ∈ χ−1
due to the same color χG(w′) = χH (wk). Combining the above two equations and noting
that disH (wk, vw) + disH (vw, uH ) ≥ disH (wk, uH ), we obtain the following result:
for any
G (χG(u))) = dropmin({{disH (wk, uH ) : uH ∈ χ−1
H (χH (u))}})
35
𝑤𝑤𝑘𝑘𝑤𝑤1𝑤𝑤𝑗𝑗𝑤𝑤𝑚𝑚𝑢𝑢𝑣𝑣𝑢𝑢𝑢𝑤𝑤𝑤𝑢𝑢𝑤𝑤𝑤𝑣𝑣𝑤𝑤𝑤𝑢𝑢1𝑤𝑢𝑢2𝑤𝑆𝑆𝑢𝑢𝑤𝑤𝑤𝑆𝑆𝑣𝑣𝑤𝑤𝑤Published as a conference paper at ICLR 2023
uH ∈ χ−1
disH (vw, uH ) = disH (wk, uH ). In particular,
H (χH (u)) for which disH (wk, vw) + disH (vw, uH ) > disG(w′, u′
w), disH (wk, vw) +
disH (wk, w1) = disH (wk, vw) + disH (vw, w1),
disH (wk, wj) = disH (wk, vw) + disH (vw, wj).
Therefore,
disH (w1, wj) = disH (w1, wk) + disH (wk, wj)
= 2disH (wk, vw) + disH (vw, w1) + disH (vw, wj)
≥ 2disH (wk, vw) + disH (w1, wj),
implying wk = vw. However, χH (wk) ∈ Cu while χH (vw) ∈ Cv, yielding a contradiction.
C.4.2 THE CASE OF χG(u) = χG(v) FOR CONNECTED GRAPHS
We first define several notations. Define the mapping fG : V → {u, v} × C as follows: fG(w) =
(hG(w), χG(w)) where
hG(w) =
(cid:26) u if disG(w, v) = disG(w, u) + 1,
if disG(w, u) = disG(w, v) + 1.
v
(12)
It is easy to see that hG is well-defined for all w ∈ V because {u, v} is a cut edge of G. We further
define the following auxiliary graph:
Definition C.34. (Auxiliary graph) Define the auxiliary graph GA = (VGA , EGA) where VGA :=
{u, v} × C and EGA := {{{fG(w1), fG(w2)}} : {w1, w2} ∈ EG}. Note that GA can have self loops,
so each edge is denoted as a multiset with two elements.
It is straightforward to see that there is only one edge in GA with the form {{(u, c1), (v, c2)}} ∈
EGA for some c1, c2 ∈ C since {u, v} is a cut edge of G. Therefore,
the only edge is
{{(u, χG(u)), (v, χG(v))}} and is a cut edge in GA.
We also define f −1
first prove that f −1
Lemma C.35. fG is a surjection.
G as the inverse mapping of fG, i.e. f −1
G is well-defined on the domain VGA .
G (z, c) = {w ∈ V : fG(w) = (z, c)}. We
G (v, c) is an empty set. Without loss of generality, assume f −1
G (u, c). Obviously, w ̸= u (otherwise f −1
Proof. Suppose that fG is not a surjection. Then there exists a color c ∈ C such that either f −1
G (u, c)
G (v, c) = ∅, then f −1
or f −1
G (u, c) ̸= ∅.
Pick any w ∈ f −1
G (v, χG(v)) = ∅, a contradiction). Then
G (v, χG(x)) is empty. Note that x ∈ f −1
we claim that for any x ∈ NG(w), f −1
G (u, χG(x)). If
the claim does not hold, take x′ ∈ f −1
G (v, χG(x)). Since x connects to a node with color c and
χG(x) = χG(x′), x′ must also connect to a node with color c. Denote the node that connects to x′
with color c as w′. Then w′ ∈ f −1
G (v, c), yielding a contradiction.
By induction, for any x such that there exists a path from x to w without going through the edge
{u, v}, we have f −1
G (v, χG(v)) = ∅, leading to a contra-
diction. Therefore, f is a surjection.
G (v, χG(x)) = ∅. This finally implies f −1
Lemma C.36. |f −1
G (u, χG(u))| = |f −1
G (v, χG(v))| = 1.
G (u,χ(u)) disG(u, u′) and similarly pick v′. It follows that any path
Proof. Pick u′ = arg maxu′∈f −1
between u′ and v′ goes through edge {u, v}. Therefore, disG(u′, v′) = disG(u, u′)+disG(v, v′)+1.
Since all nodes u, u′, v, v′ have the same color under SPD-WL, there exists a node w ∈ χ−1
G (χG(u))
satisfying disG(u, w) = disG(u′, v′) and thus disG(u, w) > disG(u, u′). By definition of the node
u′, fG(w) ̸= (u, χ(u)) and thus fG(w) = (v, χ(u)). Therefore, disG(u, w) = disG(v, w) + 1,
which implies that
disG(v, w) = disG(v, v′) + disG(u, u′).
Since disG(v, w) ≤ disG(v, v′), we have disG(v, w) = disG(v, v′) and u = u′. A similar argument
yields v = v′, finishing the proof.
36
Published as a conference paper at ICLR 2023
We can now prove some useful properties of the auxiliary graph GA based on Lemmas C.35
and C.36.
Corollary C.37. For any c1, c2 ∈ C, {{(u, c1), (u, c2)}} ∈ EGA if and only if {{(v, c1), (v, c2)}} ∈
EGA .
Proof. By definition of E A
G , if {{(u, c1), (u, c2)}} ∈ EGA , then there exists two vertices w1 ∈
f −1
G (u, c1) and w2 ∈ f −1
G (u, c2) such that {w1, w2} ∈ EG. By Lemma C.36, either χG(w1) ̸=
χG(u) or χG(w2) ̸= χG(u). Without loss of generality, assume c1 ̸= χG(u). By Lemma C.35,
there exists x1 ∈ f −1
G (v, c1). Since χG(x1) = χG(w1), x1 must also connect to a node x2
with χG(x2) = c2. The edge {x1, x2} ̸= {u, v} because χG(x1) = c1 ̸= χG(u). Therefore,
f (x2) = (v, c2), namely {{(v, c1), (v, c2)}} ∈ E A
G .
The following lemma establishes the distance relationship between graphs G and GA.
Lemma C.38. The following holds:
• For any w, w′ ∈ V, disG(w, w′) ≥ disGA (f (w), f (w′)).
• For any ξ, ξ′ ∈ V A and any node w ∈ f −1
disG(w, w′) = disGA (ξ, ξ′).
G (ξ), there exists a node w′ ∈ f −1
G (ξ′) such that
Proof. The first bullet is trivial since for all {w, w′} ∈ EG, {{f (w), f (w′)}} ∈ EGA by Defini-
tion C.34. We prove the second bullet in the following. Note that GA can have self-loops, but for
any ξ, ξ′ ∈ V A, the shortest path between ξ and ξ′ will not go through self-loops. We only need
to prove that for all {{ξ, ξ′}} ∈ E A, ξ ̸= ξ′ and all w ∈ f −1
G (ξ′) such
that {w, w′} ∈ EG. This will imply that disG(w, w′) ≤ disGA (ξ, ξ′) and completes the proof by
combining the first bullet in Lemma C.38.
The case of {{ξ, ξ′}} = {{(u, χG(u)), (v, χG(v))}} is trivial. Now assume that {{ξ, ξ′}} ̸=
{{(u, χG(u)), (v, χG(v))}}. By Definition C.34, there exists x ∈ f −1
G (ξ′), such that
{x, x′} ∈ EG. Note that hG(x) = hG(x′) because {x, x′} ̸= {u, v}. Since χG(x) = χG(w), there
exists w′ ∈ χ−1
G (χG(x′)) such that {w, w′} ∈ EG. It must hold that hG(w) = hG(w′) (otherwise
{w, w′} = {u, v} and thus {{ξ, ξ′}} = {{(u, χG(u)), (v, χG(v))). Therefore, hG(w′) = hG(w) =
hG(x) = hG(x′) and thus fG(w′) = fG(x′), namely w′ ∈ f −1
G (ξ), there exists w′ ∈ f −1
G (ξ) and x′ ∈ f −1
G (ξ′).
Lemma C.38 leads to the following corollary:
Corollary C.39. The following holds:
• For any w, w′ ∈ V satisfying χG(w) = χG(w′) and hG(w) = hG(w′) (i.e. fG(w) =
fG(w′)), disG(u, w) = disG(u, w′) and disG(v, w) = disG(v, w′);
• For any w, w′ ∈ V satisfying χG(w) = χG(w′) and hG(w) ̸= hG(w′), disG(u, w) =
disG(v, w′) and disG(v, w) = disG(u, w′).
Proof. Proof of the first bullet: by Lemma C.38, there exists two nodes u1, u2 ∈ f −1
G (fG(u)) such
that disG(u1, w) = disGA(fG(u), fG(w)) and disG(u2, w′) = disGA (fG(u), fG(w′)). Therefore,
disG(u1, w) = disG(u2, w′). However, by Lemma C.36 and the condition hG(w) = hG(w′), it
must be u1 = u2 = u, namely disG(u, w) = disG(u, w′). The proof of disG(v, w) = disG(v′, w′)
is similar.
Proof of the second bullet: Let χG(w) = χG(w′) = c. Without loss of generality, assume fG(w) =
(u, c) and f (w′) = (v, c). By Lemma C.38, it suffices to prove that disGA((u, χG(u)), (u, c)) =
disGA ((v, χG(v)), (v, c)). By the definition of GA and its cut edge {{(u, χG(u), (v, χG(v))}}, the
shortest path between (u, χG(u)) and (u, c) must only go through nodes in the set {(u, c1) : c1 ∈ C},
and similarly the shortest path between (v, χG(v)) and (v, c) must only go through nodes in
{(v, c2) : c2 ∈ C}. Finally, Corollary C.37 says that for c1, c2 ∈ C, {{(u, c1), (u, c2)}} ∈ GA
if and only if {{(v, c1), (v, c2)}} ∈ GA. We thus conclude that disGA ((u, χG(u)), (u, c)) =
disGA ((v, χG(v)), (v, c)) and disG(u, w) = disG(v, w′).
Finally, we can prove the following important corollary:
37
Published as a conference paper at ICLR 2023
Corollary C.40. For any c ∈ C, |f −1
G (u, c)| = |f −1
G (v, c)|.
Proof. Pick any w ∈ f −1
G (u, c) and x ∈ f −1
G (v, c). By Corollary C.39, we have
disG(w, u) = disG(x, v) := d,
disG(w, v) = disG(x, u) = d + 1.
The multiset {{disG(u, w′) : χG(w′) = c}} contains |f −1
elements of value d + 1. The multiset {{disG(v, w′) : χG(w′) = c}} has |f −1
value d and |f −1
the two multiset must be equivalent. Therefore, |f −1
G (v, c)|
G (v, c)| elements of
G (u, c)| elements of value d + 1. Since u and v has the same color under SPD-WL,
G (u, c)| elements of value d and |f −1
G (u, c)| = |f −1
G (v, c)|.
Next, we switch our focus to the graph H. Since we have assumed that the graph representations
of G and H are the same, i.e. {{χG(w) : w ∈ V}} = {{χH (w) : w ∈ V}}, the size of the set
{w ∈ V : χH (w) = χG(u)} must be 2. We may denote the elements as u and v without abuse of
notation and thus {u, v} ∈ EH . Also for any w ∈ V, we have disH (w, u) ̸= disH (w, v). Therefore,
we can similarly define the mapping fH : V → {u, v} × V and the mapping hH : V → {u, v} as in
(12). The auxiliary graph H A is defined analogous to Definition C.34.
Lemma C.41. For any c ∈ C, |f −1
H (u, c)| = |f −1
H (v, c)| = |f −1
G (u, c)| = |f −1
G (v, c)|.
H (u, c)| ̸= |f −1
Proof. If |f −1
H (v, c)|, we have {{disH (u, w) : χH (w) = c}} ̸= {{disH (v, w) :
χH (w) = c}}, implying that u and v cannot have the same color under SPD-WL. This already
concludes the proof by using Corollary C.40 as
|f −1
H (u, c)| + |f −1
H (v, c)| = |f −1
G (u, c)| + |f −1
G (v, c)|.
We finally present a technical lemma which will be used in the subsequent proof.
Lemma C.42. Given node w ∈ V and color c ∈ C, define multisets
DG,=(w, c) := {{disG(w, x) : x ∈ χ−1
DG,̸=(w, c) := {{disG(w, x) : x ∈ χ−1
G (c), hG(x) = hG(w)}},
G (c), hG(x) ̸= hG(w)}}.
For any two nodes w, w′ ∈ V in graphs G and H satisfying χG(w) = χH (w′), pick any d ∈
DG,̸=(w, c) and d′ ∈ DH,=(w′, c). Then d′ < d.
Proof. Without loss of generality, assume hG(w) = hH (w′) = u and let fG(w) = fH (w′) =
(u, cw). Pick x ∈ f −1
H (u, c), then disH (x′, u) = min(disG(x, u), disG(x, v))
and disH (w′, u) = min(disG(w, u), disG(w, v)). Thus
disH (w′, x′) ≤ disH (w′, u) + disH (u, x′)
G (v, c) and x′ ∈ f −1
= min(disG(w, u), disG(w, v)) + min(disG(x, u), disG(x, v))
< min(disG(w, u) + disG(x, v), disG(w, v) + disG(x, u)) + 1
= disG(w, x),
which concludes the proof.
In the following, we will prove that {u, v} is a cut edge in graph H. Consider an edge
{{(u, c1), (v, c2)}} ∈ EH A (such an edge exists because {{(u, χH (u)), (v, χH (v))}} ∈ E A
H ). We
will prove that this is the only case, i.e. it must be c1 = χH (u) = χH (v) = c2.
By Definition C.34, {{(u, c1), (v, c2)}} ∈ EH A implies that there exists two nodes x′ ∈ f −1
and w′ ∈ f −1
H (v, c2), such that {w′, x′} ∈ EH . Pick w ∈ χ−1
DG,̸=(w, c1) = ∅. Since w′ and w have the same color under SPD-WL,
H (u, c1)
G (c2). By Lemma C.42, DH,=(w′, c1)∩
DH,=(w′, c1) ∪ DH,̸=(w′, c1) = DG,=(w, c1) ∪ DG,̸=(w, c1).
By Lemma C.41, |DH,=(w′, c1)| = |DH,̸=(w′, c1)| = |DG,=(w, c1)| = |DG,̸=(w, c1)|. Therefore,
DG,̸=(w, c1) = DH,̸=(w′, c1). We claim that all elements in the set DG,̸=(w, c1) are the same. This
is because for any x ∈ χ−1
G (c1), hG(x) ̸= hG(w), we have
disG(w, x) = disG(w, h(w)) + 1 + disG(h(x), x),
38
Published as a conference paper at ICLR 2023
and by Corollary C.39 disG(w, h(w)) (or disG(h(x), x)) has an equal value for different x. Since
{w′, x′} ∈ EH , we have 1 ∈ DH,̸=(w′, c1) and thus all elements in DG,̸=(w, c1) equals 1. There-
fore, c1 = χG(u). Analogously, c2 = χG(u). Therefore, c1 = χH (u) = χH (v) = c2.
Let Su = {w ∈ V : hH (w) = u} and Sv = {w ∈ V : hH (w) = v}. Then the above argument
implies that if w ∈ Su, x ∈ Sv and {w, x} ∈ EG, then {w, x} = {u, v}. Therefore {u, v} is a cut
edge of graph H.
C.4.3 THE GENERAL CASE
The above proof assumes that the graphs G and H are both connected, and their graph representa-
tions are euqal, i.e. {{χG(w) : w ∈ V}} = {{χH (w) : w ∈ V}}. In the subsequent proof we remove
these assumptions and prove the general setting.
Lemma C.43. Either of the following two properties holds:
• {{χG(w) : w ∈ V}} = {{χH (w) : w ∈ V}};
• {{χG(w) : w ∈ V}} ∩ {{χH (w) : w ∈ V}} = ∅.
Proof. Consider the GD-WL procedure defined in Algorithm 4 with arbitrary distance function dG.
Suppose at iteration t ≥ T , {{χt
H (w) : w ∈ V}}. Then at iteration t + 1, we
have for each v ∈ V,
G(w) : w ∈ V}} ̸= {{χt
Therefore, χt+1
G (v) ̸= χt+1
G (v) = hash (cid:0){{hash(dG(v, u), χt
χt+1
G (u) for all u, v ∈ V, namely
G (w) : w ∈ V}} ∩ {{χt+1
{{χt+1
H (w) : w ∈ V}} = ∅.
G(u)) : u ∈ V}}(cid:1) .
Finally, by the injective property of the hash function, for any t ≥ T + 1, the above equation always
holds. Therefore, the stable color mappings χG and χH satisfy Lemma C.43.
The above lemma implies that if there exists edges {w1, w2} ∈ EG, {x1, x2} ∈ EH satisfying
{{χG(w1), χG(w2)}} = {{χH (x1), χH (x2)}}, then {{χG(w) : w ∈ V}} = {{χH (w) : w ∈ V}}.
Also, SPD-WL ensures that both graphs are either connected or disconnected. If they are both con-
nected, the previous proof (Appendices C.4.1 and C.4.2) ensures that {w1, w2} is a cut edge of G if
and only if {x1, x2} is a cut edge of H. For the disconnected case, let SG ⊂ V be the largest con-
nected component containing nodes w1, w2, and similarly denote SH ⊂ V as the largest connected
component containing nodes x1, x2. Obviously, |SG| = |SH | due to the facts that disG(w1, y) =
∞ ̸= disG(w1, y′) for all y /∈ SG, y ∈ SG and that the two edges {w1, w2} ∈ EG, {x1, x2} have
the same color under SPD-WL. Moreover, {{χG(w) : w ∈ SG}} = {{χH (w) : w ∈ SH }}. Now
consider re-execute the SPD-WL algorithm on subgraphs G[SG] and H[SH ] induced by the vertices
in set SG and SH , respectively. It follows that for any uG ∈ SG and uH ∈ SH , χG(uG) = χH (uH )
implies that χG[SG](uG) = χH[SH ](uH ). Therefore, {w1, w2} is a cut edge of G[SG] if and only if
{x1, x2} is a cut edge of H[SH ]. By the dinifition of SG and SH , {w1, w2} is a cut edge of G if and
only if {x1, x2} is a cut edge of H.
It remains to prove that {{χG(w) : w ∈ V}} = {{χH (w) : w ∈ V}} implies BCETree(G) ≃
BCETree(H). By definition of the block cut-edge tree, each cut edge of G corresponds to
a tree edge in BCETree(G) and each biconnected component of G corresponds to a node of
BCETree(G). We still only focus on the case of connected graphs G, H, and it is straightfor-
ward to extend the proof to the general (disconnected) case using a similar technique as the previous
paragraph.
Given a fixed SPD-WL graph representation R, consider any graphs G = (V, EG) satisfying
{{χG(w) : w ∈ V}} = R. Since we have proved that the SPD-WL node feature χG(v), v ∈ V
precisely locates all the cut edges, the multiset
CE := {{{χG(u), χG(v)} : {u, v} ∈ EG is a cut edge}}
is fixed (fully determined by R, not G). Denote CV := (cid:83)
{c1,c2}∈CE {c1, c2} as the set that contains
the color of endpoints of all cut edges. For each cut edge {u, v} ∈ EG, denote SG,u and SG,v be
39
Published as a conference paper at ICLR 2023
the vertex partition corresponding to the two connected components after removing the edge {u, v},
satisfying u ∈ SG,u, v ∈ SG,v, SG,u ∩ SG,v = ∅, SG,u ∪ SG,v = V. It suffices to prove that given a
cut edge {u, v} ∈ EG with color {χG(u), χG(v)}, the multiset {{χG(w) : w ∈ SG,u, χG(w) ∈ CV}}
can be determined purely based on R and the edge color {{χG(u), χG(v)}}, rather than the specific
graph G or edge {u, v}. This basically concludes the proof, since the BCETree can be uniquely
if {{χG(w) : w ∈ SG,u, χG(w) ∈ CV}} = {{χG(u)}} (i.e. with only
constructed as follows:
one element), then {{χG(u), χG(v)}} is a leaf edge of the BCETree such that χG(u) connects to a
biconnected component that is a leaf of the BCETree. After finding all the leaf edges, we can then
find the BCETree edges that connect to leaf edges and determine which leaf edges they connect. The
procedure can be recursively executed until the full BCETree is constructed. The whole procedure
does not depend on the specific graph G and only depends on R.
We now show how to determine {{χG(w) : w ∈ SG,u, χG(w) ∈ CV}} given a cut edge {u, v} ∈ EG
with color {χG(u), χG(v)}. Define the multiset
D(c1, c2) := {{disG(w, x) : x ∈ χ−1
G (c2)}}
(w ∈ χ−1
G (c1) can be picked arbitrarily)
Note that D(c1, c2) is well-defined (does not depend on w) by definition of the SPD-WL color. For
any cu, cv ∈ CE, pick arbitrary cut edge {u, v} with color χG(u) = cu, χG(v) = cv. Define
T (cu, cv) =
(cid:91)
c∈CV
{{c}} × |(D(cu, c) + 1) ∩ D(cv, c)|
(13)
where {{c}} × m denotes a multiset with m repeated elements c, and D(cu, c) + 1 := {{d + 1 :
d ∈ D(cu, c)}}. Intuitively speaking, T (cu, cv) corresponds to the color of all nodes w ∈ V such
that disG(u, w) + 1 = disG(v, w) and χG(w) ∈ CV. Therefore, T (cu, cv) is exactly the multiset
{{χG(w) : w ∈ SG,u, χG(w) ∈ CV}} and we have completed the proof.
C.5 PROOF OF THEOREM 4.2
Theorem C.44. Let G = (V, EG) and H = (V, EH ) be two graphs, and let χG and χH be the
corresponding RD-WL color mapping. Then the following holds:
• For any two nodes w ∈ V in G and x ∈ V in H, if χG(w) = χH (x), then w is a cut vertex
of G if and only if x is a cut vertex of H.
• If the graph representations of G and H are the same under RD-WL, then their block
cut-vertex trees (Definition 2.4) are isomorphic. Mathematically, {{χG(w) : w ∈ V}} =
{{χH (w) : w ∈ V}} implies that BCVTree(G) ≃ BCVTree(H).
Proof Sketch. First observe that Lemma C.43 holds for general distances and thus applies here.
Therefore, if χG(w) = χH (x), the graph representations will be the same, i.e. {{χG(w) : w ∈
V}} = {{χH (w) : w ∈ V}}. By a similar analysis as SPD-WL (Appendix C.4.3), we can only
focus on the case that both graphs are connected. We prove the first bullet of Theorem 4.2 in
Appendix C.5.1 and prove the second bullet in Appendix C.5.2, both assuming that G and H are
connected and their graph representations are the same.
C.5.1 PROOF OF THE FIRST PART
We first present a key property of the Resistance Distance, which surprisingly relates to the cut
vertices in a graph.
Lemma C.45. Let G = (V, E) be a connected graph and v ∈ V. Then v is a cut vertex of G if
and only if there exists two nodes u, w ∈ V, u ̸= v, w ̸= v, such that disR
G(v, w) =
disR
G(u, v) + disR
G(u, w).
Proof. We use the key finding that the Resistance Distance is equivalent to the Commute Time Dis-
tance multiplied by a constant (Chandra et al., 1996, see also Appendix E.2), i.e. disC
G(u, w) =
2|E| disR
G(u, w) := hG(u, w) +
hG(w, u) where hG(u, w) is the average hitting time from u to w in a random walk (Appendix E.2).
G(u, w). Here, the Commute Time Distance is defined as disC
40
Published as a conference paper at ICLR 2023
all hitting paths Puw from u to w (not necessarily simple) into two sets P v
that all paths P ∈ P v
and P
1/ (cid:81)m−1
• If v is not a cut vertex, given any nodes u, w, u ̸= v, w ̸= v, we can partition the set of
v
uw such
uw ̸= ∅
v
uw ̸= ∅. Given a path P = (x0, · · · , xm), define the probability function q(P ) :=
i=0 degG(xi). Then by definitions of the average hitting time h,
(cid:88)
uw and P
v
uw contains v. Clearly, P v
uw contain v and no path P ∈ P
(cid:88)
(cid:88)
q(P ) · |P | =
q(P ) · |P | +
q(P ) · |P |
hG(u, w) =
=
=
≤
<
P ∈P v
uw
P ∈Puw
(cid:88)
P1∈P w
uv,P2∈Pvw
P ∈P v
uw
(cid:88)
P ∈P v
uw
q(P1)q(P2)(|P1| + |P2|) +
q(P ) · |P |
(cid:88)
P1∈P w
uv
(cid:88)
+
q(P1)|P1|
(cid:32)
(cid:88)
(cid:33)
q(P2)
+
(cid:88)
q(P2)|P2|
P2∈Pvw
P2∈Pvw


(cid:88)
P1∈P w
uv

q(P1)

q(P )|P |
P ∈P v
(cid:88)
uw
q(P )|P | +
P ∈P w
uv
(cid:88)
P ∈P w
uv
q(P )|P | +
(cid:88)
P ∈P v
(cid:88)
uw
P ∈P w
uv
q(P )|P | +
(cid:88)
q(P )|P |
q(P )|P | +
P ∈Pvw
(cid:88)
P ∈Pvw
q(P )|P |
= hG(u, v) + hG(v, w).
We can similarly prove that hG(w, u) < hG(w, v) + hG(v, u).
• If v is a cut vertex, then there exists two different nodes u, w ∈ V, u ̸= v, w ̸= v, such
that any path from u to w goes through v. A similar analysis yields the conclusion that
hG(u, w) = hG(u, v) + hG(v, w) and hG(w, u) = hG(w, v) + hG(v, u).
This completes the proof of Lemma C.45.
In the subsequent proof, assume u ∈ V is a cut vertex of G, and let {Si}m
i=1 (m ≥ 2) be the partition
of the vertex set V\{u}, representing each connected component after removing node u. We have
the following lemma (which has a similar form as Lemma C.27):
Lemma C.46. There is at most one set Si satisfying Si ∩ χ−1
Si ∩ χ−1
G (χG(u)) ̸= ∅ for some i ∈ [m], then for any j ∈ [m] and j ̸= i, Sj ∩ χ−1
G (χG(u)) ̸= ∅. In other words, if
G (χG(u)) = ∅.
G (χG(u)) = ∅.
G (χG(u)) disR
G(u, u′). If ui = u, then Si ∩ χ−1
Proof. Let ui = arg maxu′∈χ−1
G (χG(u)) = ∅ for all
i ∈ [m] and thus Lemma C.46 clearly holds. Otherwise, ui ∈ Si for some i. We will prove that for
any j ̸= i, Sj ∩ χ−1
If the above conclusion does not holds, then we can pick a set Sj and a vertex uj ∈ Sj ∩χ−1
G (χG(u)).
Since u is a cut vertex and Si, Sj are different connected components, by Lemma C.45 we have
disR
G(ui, uj) = disR
G(ui, u). This yields a contradiction because
G (χG(u)) disR
G(ui, u′), which means that u and ui can-
maxu′∈χ−1
not have the same RD-WL color.
G(ui, u) + disR
G(u, u′) ̸= maxu′∈χ−1
G (χG(ui)) disR
G(u, uj) > disR
The next lemma presents a key result which is similar to Corollary C.30.
Lemma C.47. For all u′ ∈ χ−1
G (χG(u)), u′ it is a cut vertex of G.
Proof. If |χ−1
exists two sets Si and Sj satisfying Si ∩ χ−1
G (χG(u))| = 1, then Lemma C.47 clearly holds. Otherwise, by Lemma C.46 there
G (χG(u)) ̸= ∅, Sj ∩ χ−1
G (χG(u)) = ∅. Since Sj ̸= ∅, we
41
Published as a conference paper at ICLR 2023
can pick w ∈ Sj with color χG(w) ̸= χG(u). Pick u′ ∈ Si ∩ χ−1
there exists a node w′ ∈ χ−1
G(u, w) = disR
G (χG(u)). Since χG(u) = χG(u′),
G(u′, w′). Then we have
{{disR
G(w, u′′) : u′′ ∈ χ−1
G (χG(u))}} (14)
G (χG(u))}} (15)
where (14) holds because u is a cut vertex and all u′′ ̸= u are in the set Si but w ∈ Sj (Lemma C.46),
and (15) holds because χG(u) = χG(u′). On the other hands,
G(u, u′′) : u′′ ∈ χ−1
G(u′, u′′) : u′′ ∈ χ−1
G(w, u) + disR
G(w′, u′) + disR
G (χG(w)) such that disR
G (χG(u))}} = {{disR
= {{disR
{{disR
G(w, u′′) : u′′ ∈ χ−1
G(w′, u′′) = disR
G (χG(u))}} = {{disR
G(w′, u′′) : u′′ ∈ χ−1
G (χG(u))}}.
Therefore, disR
G (χG(u)). Pick u′′ = u,
then clearly u′′ ̸= u′ and u′′ ̸= w. Lemma C.45 shows that u′ is a cut vertex, which concludes the
proof. See Figure 10 for an illustration of the above proof.
G(u′, u′′) for all u′′ ∈ χ−1
G(w′, u′) + disR
H ∈ χ−1
H (χG(u)), u′
H (χG(u)) satisfying disR
G(w,u). Pick another node u′
Using a similar proof technique as the one in Lemma C.47, we
can prove the first bullet of Theorem 4.2. Note that we have as-
sumed {{χG(w) : w ∈ V}} = {{χH (w) : w ∈ V}}. First consider
the case when |χ−1
G (χG(u))| > 1. Pick wH ∈ χ−1
H (χG(w))
where w is defined in the proof of Lemma C.47.
Then,
there exists uH ∈ χ−1
H (wH , uH ) =
disR
H ̸= uH
(this is feasible as |χ−1
H (χG(u))| > 1). Following the proce-
dure of the above proof, we can obtain that disR
H (wH , u′′) =
disR
H (χG(u)). There-
H ) = disR
fore, disR
H (uH , u′
H ), imply-
ing uH is a cut vertex of H by Lemma C.45.
Now consider the case when |χ−1
node in χ−1
S2, then disR
in H, then there exists a node w′
also have disR
H (w′
with color χG(u) in H. Therefore, disR
in H (Lemma C.45).
G(u, w2) = disR
2 ∈ χ−1
G(w1, u) and disR
H (uH , u′′) for all u′′ ∈ χ−1
H (wH , uH ) + disR
H (wH , uH )+disR
H (wH , u′
G (χG(u))| = 1. Then |χ−1
H (w′
1, u)+disR
G(w1, u) + disR
H (χG(u))| = 1 and we can denote the
H (χG(u)) as u without abuse of notation. Choose arbitrary two nodes w1 ∈ S1 and w2 ∈
H (χG(w1))
1, w′
2). We
G(w2, u) because u is the unique node
2) = disR
2) and u is a cut vertex
G(w1, w2) (Lemma C.45). Pick any w′
H (χG(w2)) satisfying disR
2, u) = disR
H (u, w′
G(w1, w2) = disR
1 ∈ χ−1
H (w′
1, u) = disR
H (w′
H (w′
1, w′
Figure 10:
proof of Lemma C.47.
Illustration of the
C.5.2 PROOF OF THE SECOND PART
i=1
We first introduce some notations. As before, we assume G and H are connected and {{χG(w) :
w ∈ V}} = {{χH (w) : w ∈ V}}. As we will consider multiple cut vertices in the following proof, we
adopt the notation {SG,i(u)}mG(u)
, which represents the set of connected components of graph G
after removing node u. Here, mG(u) is the number of connected components after removing node
u, which is greater than 1 if u is a cut vertex. It follows that (cid:83)mG(u)
SG,i(u) = V\{u}. We further
i=1
define the index set MG(u) := {i ∈ [mG(u)] : SG,i(u) ∩ χ−1
G (χG(u)) = ∅}. By Lemma C.46,
either |MG(u)| = mG(u) − 1 or |MG(u)| = mG(u).
Lemma C.48. Let u ∈ V be a cut vertex of G. Let u′ ∈ χ−1
H (χG(u)), then u′ is also a cut
vertex of H. Let i ∈ [mG(u)] and j ∈ [mH (u′)] be two indices and pick nodes w ∈ SG,i(u) and
w′ ∈ SH,j(u′). Assume w and w′ have the same color, i.e. χG(w) = χH (w′). Then the following
holds:
• If i ∈ MG(u) and j ∈ MH (u′), then disR
• If i ∈ MG(u) and j /∈ MH (u′), then disR
G(w, u) = disR
G(w, u) < disR
H (w′, u′).
H (w′, u′).
Proof. Proof of the first bullet: since i ∈ MG(u), any path from w to a node uG ∈ χ−1
goes through the cut vertex u, implying minuG∈χ−1
G(w, uG) = disR
G (χG(u))
G(w, u). Similarly,
G (χG(u)) disR
42
𝑢𝑢𝑆𝑆𝑖𝑖𝑢𝑢′𝑤𝑤𝑤𝑤′𝑆𝑆𝑗𝑗Published as a conference paper at ICLR 2023
H (χH (u′)) disR
since j ∈ MH (u′), minuH ∈χ−1
and w′ are the same under RD-WL, we have
H (w′, uH ) = disR
H (w′, u′). Since the color of nodes w
min
H (χH (u′))
uH ∈χ−1
disR
H (w′, uH ) =
min
G (χG(u))
uG∈χ−1
disR
G(w, uG)
and thus disR
H (w′, u′) = disR
G(w, u).
Proof of the second bullet: first note that disR
H (w′, u′) ≥ disR
G(w, u) because
disR
H (w′, u′) ≥
min
H (χH (u′))
uH ∈χ−1
disR
H (w′, uH ) =
min
G (χG(u))
uG∈χ−1
disR
G(w, uG) = disR
G(w, u).
If the lemma does not hold, then disR
H (w′, u′) = disR
G(w, u). Consequently,
{{disR
G(w, uG) : uG ∈ χ−1
G (χG(u))}} = {{disR
= {{disR
G(w, u) + disR
H (w′, u′) + disR
G(u, uG) : uG ∈ χ−1
H (u′, uH ) : uH ∈ χ−1
G (χG(u))}}
H (χH (u′))}}.
On the other hands,
{{disR
G(w, uG) : uG ∈ χ−1
H (w′, uH ) = disR
G (χG(u))}} = {{disR
H (w′, u′) + disR
H (w′, uH ) : uH ∈ χ−1
H (u′, uH ) for all uH ∈ χ−1
H (χH (u′))}}.
H (χH (u′)). However,
H (w′, u′′) <
H (u′, u′′) because w′ and u′′ are in the same connected component (Lemma C.45).
Therefore, disR
we can choose u′′ ∈ χ−1
disR
This yields a contradiction and concludes the proof.
Corollary C.49. Let u ∈ V be a cut vertex of G. Let u′ ∈ χ−1
H (χG(u)), then u′ is also a cut vertex
of H. Pick any SG,i(u) and SH,j(u′) with indices i ∈ MG(u) and j ∈ MH (u′). Then either of the
following holds:
H (χH (u′)) ∩ SH,j(u′) by definition of j, and clearly disR
H (w′, u′)+disR
• {{χG(w) : w ∈ SG,i(u)}} = {{χH (w) : w ∈ SH,j(u′)}}.
• {{χG(w) : w ∈ SG,i(u)}} ∩ {{χH (w) : w ∈ SH,j(u′)}} = ∅.
Proof. Assume {{χG(w) : w ∈ SG,i(u)}} ∩ {{χH (w) : w ∈ SH,j(u′)}} ̸= ∅. Then there exists
nodes w ∈ SG,i(u) in G and w′ ∈ SH,j(u′) in H, satisfying χG(w) = χH (w′). Our goal is to
prove that {{χG(w) : w ∈ SG,i(u)}} = {{χH (w) : w ∈ SH,j(u′)}}. It thus suffices to prove that for
any color c ∈ C, |χ−1
H (c) ∩ SH,j(u′)|.
G (c) ∩ SG,i(u)| = |χ−1
Define DG(w, c) = {{disR
DG(w, c)}}. We next claim that
G(w, x) : x ∈ χ−1
G (c)}} and define DG(w, c) + d := {{d + d′ : d′ ∈
|χ−1
G (c) ∩ SG,i(u)| = |χ−1
This is simply because for any x ∈ χ−1
then disR
disR
G(w, x) = disR
G(w, u) + disR
G(u, x). Similarly,
|χ−1
G (c)| − |DG(w, c) ∩ (DG(u, c) + disR
G (c), either x ∈ SG,i(u) or x /∈ SG,i(u). If x /∈ SG,i(u),
G(w, u) +
G(u, x) (Lemma C.45); otherwise, disR
G(w, x) ̸= disR
G(w, u))|.
H (w′, u′))|.
G(w, u) = disR
H (c) ∩ SH,j(u′)| = |χ−1
H (c)| − |DH (w′, c) ∩ (DH (u′, c) + disR
Noting that |χ−1
G (c)| = |χ−1
(Lemma C.48), we obtain |χ−1
H (c)|, DG(w, c) = DH (w′, c), and disR
G (c) ∩ SG,i(u)| = |χ−1
H (c) ∩ SH,j(u′)| and conclude the proof.
H (w′, u′)
Remark C.50. As a special case, Lemma C.48 and Corollary C.49 also hold when G = H. For ex-
ample, Corollary C.49 implies that for any SG,i(u) and SG,j(u) such that SG,i(u) ∩ χ−1
G (χG(u)) =
SG,j(u) ∩ χ−1
G (χG(u)) = ∅, either of the two items in Corollary C.49 holds.
Lemma C.48 and Corollary C.49 leads to the following key corollary:
Corollary C.51. Let u ∈ V be a vertex in G and u′ ∈ V be a vertex in H. If χG(u) = χH (u′), then
mG(u) = mH (u′) and
{{{{χG(w) : w ∈ SG,i(u)}}}}mG(u)
i=1 = {{{{χH (w) : w ∈ SH,i(u′)}}}}mH (u′)
i=1
.
43
Published as a conference paper at ICLR 2023
Proof. If both u and u′ are not cut vertices, Corollary C.51 trivially holds since mG(u) = mH (u′) =
1 and SG,1(u) = V\{u}, SH,1(u′) = V\{u′}. Now assume u and u′ are both cut vertices. We first
claim that
{{χG(w) : w ∈ (cid:83)
i∈MG(u) SG,i(u)}} = {{χH (w) : w ∈ (cid:83)
i∈MH (u′) SH,i(u′)}}.
(16)
To prove the claim, it suffices to prove that for each color c ∈ C,
(cid:12)
(cid:12)
(cid:12)
(cid:12)
(cid:12)
(cid:12)
(cid:91)
SG,i(u) ∩ χ−1
G (c)
i∈MG(u)
(cid:12)
(cid:12)
(cid:12)
(cid:12)
(cid:12)
(cid:12)
=
(cid:12)
(cid:12)
(cid:12)
(cid:12)
(cid:12)
(cid:12)
(cid:91)
i∈MH (u′)
SH,i(u′) ∩ χ−1
(cid:12)
(cid:12)
(cid:12)
H (c)
(cid:12)
(cid:12)
(cid:12)
.
(17)
G (c)| = |χ−1
Note that |χ−1
(cid:83)
i /∈MG(u) SG,i(u) ∩ χ−1
i∈MG(u) SG,i(u) ∩ χ−1
G(u, w2). In other words, the following two sets does not intersect:
disR
H (c)|. Also note that by Lemma C.48, for any two nodes w1 ∈
G(u, w1) <
G (c), we have disR
G (c) and w2 ∈ (cid:83)
DG(u, c) := {{disR
(cid:101)DG(u, c) := {{disR
G(w, u) : w ∈ (cid:83)
G(w, u) : w ∈ (cid:83)
i∈MG(u) SG,i(u) ∩ χ−1
i /∈MG(u) SG,i(u) ∩ χ−1
G (c)}},
G (c)}}.
Since χG(u) = χH (u′), we have DG(u, c) ∪ (cid:101)DG(u, c) = DH (u′, c) ∪ (cid:101)DH (u′, c). Then DG(u, c) ∩
(cid:101)DG(u, c) = DH (u′, c) ∩ (cid:101)DH (u′, c) = ∅ implies that DG(u, c) = DH (u′, c) and (cid:101)DG(u, c) =
(cid:101)DH (u′, c). This proves (17) and thus (16) holds.
We next claim that
{{{{χG(w) : w ∈ SG,i(u)}} : i ∈ MG(u)}} = {{{{χH (w) : w ∈ SH,i(u′)}} : i ∈ MH (u′)}}. (18)
This simply follows by using (16) and Corollary C.49. Finally, (18) already yields the desired
conclusion because:
• If |MG(u)| = mG(u), then (16) implies that
(cid:12)
(cid:12)
(cid:12)
(cid:12)
(cid:12)
i∈MH (u′) SH,i(u′)
(cid:12) =
(cid:83)
(cid:12)
(cid:12)
(cid:12)
(cid:83)
i∈MG(u) SG,i(u)
(cid:12)
(cid:12)
(cid:12) = |V| − 1
and thus |MH (u′)| = mH (u′).
• If |MG(u)| = mG(u) − 1, then analogously |MH (u′)| = mH (u) − 1. Furthermore,
{{{{χG(w) : w ∈ SG,i(u)}} : i /∈ MG(u)}} = {{{{χG(w) : w ∈ SH,i(u′)}} : i /∈ MH (u′)}}
because {{χG(w) : w ∈ V\{u}}} = {{χH (w) : w ∈ V\{u′}}}.
In both cases, Corollary C.51 holds.
We are now ready to prove that {{χG(w) : w ∈ V}} = {{χH (w) : w ∈ V}} implies BCVTree(G) ≃
BCVTree(H). Recall that in a block cut-vertex tree BCVTree(G), there are two types of nodes: all
cut vertices of G, and all biconnected components of G. Each edge in BCVTree(G) is connected
between a cut vertex u ∈ V and a biconnected component B ⊂ V such that u ∈ B.
Given a fixed RD-WL graph representation R, consider any graph G = (V, EG) satisfying
{{χG(w) : w ∈ V}} = R. First, all cut vertices of G can be determined purely from R using the
node colors. We denote the cut vertex color multiset as CV := {{χG(u) : u is a cut vertex of G}}.
Next, the number mG(u) for each cut vertex u can be determined only by its color χG(u) (by Corol-
lary C.51), which is equal to the degree of node u in BCVTree(G). We now give a procedure to
construct BCVTree(G), which purely depends on R rather than the specific graph G.
We examine the multisets T (u) := {{{{χG(w) : w ∈ SG,i(u)}}}}mG(u)
for all cut vertices u, which
only depends on R and χG(u) rather than the specific graph G or node u by Corollary C.51. See
Figure 11(b) for an illustration of Tu for four types of cut vertices u. In the first step, we find all
cut vertices u such that (cid:80)
In
other words, we find cut vertices u such that there is at most one connected component SG,i(u) that
contains cut vertices. These cut vertices u will serve as “leaf (cut vertex) nodes” in BCVTree(G), in
the sense that it connects to at most one internal node in BCVTree(G). The number of BCVTree leaf
S∈T (u) 1[CV ∩ S ̸= ∅] ≤ 1 where 1[·] is the indicator function.
i=1
44
Published as a conference paper at ICLR 2023
(a) The original graph
(b) Illustration of the multisets T (u) for each cut vertex u.
(c) The first step
(d) The second step
(e) The third step
(f) The final step
Figure 11: Illustrations for constructing the BCVTree given the graph representation R.
nodes that connect to u are also determined by Corollary C.51. See Figure 11(c) for an illustration.
After finding all the “leaf (cut vertex) nodes”, we can then find cut vertex nodes v such that when
removing all “leaf (cut vertex) nodes” in the BCVTree, v will serve as a “leaf (cut vertex) node”. To
do this, we compute for each cut vertex v and each biconnected component Bv associated with v,
whether Bv has no cut vertex or all cut vertices in Bv correspond to the “leaf (cut vertex) nodes” in
BCVTree(G). Then, we check whether a cut vertex v satisfies (cid:80)
u ̸= ∅] ≤ 1,
where the set CV
u contains all colors corresponding to “leaf (cut vertex) nodes”. These vertices v will
serve as new “leaf (cut vertex) nodes” when removing all “leaf (cut vertex) nodes” in the BCVTree,
and the connection between such vertices v and “leaf (cut vertex) nodes” can also be determined (see
Figure 11(d) for an illustration). The procedure can be recursively executed until the full BCVTree
is constructed (see Figure 11(f)), and the whole procedure does not depend on the specific graph G
and only depends on R, which completes the proof.
S∈T (v) 1[(CV ∩ S)\CV
C.6 PROOF OF THEOREM 4.5
Given a graph G = (V, E), let χt
G be the 2-FWL color mapping after the t-th iteration (see Algo-
rithm 2 for details), and let χG be the stable 2-FWL color mapping. The following result is useful
for the subsequent proof:
Lemma C.52. Let u1, u2, v1, v2 ∈ V be nodes in graph G and t be an integer. The following holds:
• If χt
G(u1, v1) = χt
G(u2, v2), then u1 = v1 if and only if u2 = v2;
• If χt
G(u1, v1) = χt
G(u2, v2), then {u1, v1} ∈ E if and only if {u2, v2} ∈ E;
• If χt
G(u1, v1) = χt
G(u2, v2) and t ≥ 1, then degG(u1) = degG(u2) and degG(v1) =
degG(v2).
45
×3×4×2×2×3×4×2×2×2×2×1×2×2×2×1×2×2×1×1Published as a conference paper at ICLR 2023
Proof. By the initial coloring (6) of 2-FWL, χ0
G(u, v) can have the following three types of values:
χ0
G(u1, v1) =
(cid:40) csame
cedge
cother
if u = v
if u ̸= v and {u, v} ∈ E
if u ̸= v and {u, v} /∈ E
where csame, cedge, cother are three different colors. Therefore, if χ0
u1 = v1 if and only if u2 = v2, and {u1, v1} ∈ E if and only if {u2, v2} ∈ E. For the update step,
G(u1, v1) = χ0
G(u2, v2), then
G(u, v) = hash (cid:0)χt−1
χt
G (u, v), {{(χt−1
G (u, w), χt−1
G (w, v)) : w ∈ V}}(cid:1) .
G(u1, w) : w ∈ V}} = {{χ0
(19)
G(u1, v1) = χ1
G(u2, v2), then (19) implies that {{χ0
If χ1
G(u2, w) : w ∈
V}} and thus |{{w ∈ V : {u1, w} ∈ E}}| = |{{w ∈ V : {u2, w} ∈ E}}|, namely degG(u1) =
degG(u2). We can similarly prove that degG(v1) = degG(v2).
Finally, note that χt
concludes the proof of the case t ≥ 1 by a simple induction.
G (u2, v2) using (19). This
G(u2, v2) implies χt−1
G (u1, v1) = χt−1
G(u1, v1) = χt
For a path P = (x0, · · · , xd) (not necessarily simple) in graph G of length d ≥ 1, define ω(P ) :=
(degG(x1), · · · , degG(xd−1)) which is a tuple of length d − 1. We have the following key lemma:
Lemma C.53. Let t ∈ N be a non-negative integer. Given nodes u1, u2, v1, v2 ∈ V, if χt
G(u1, v1) =
χt
G(u2, v2), then the following holds:
• Denote Pd(u, v) be the set of all paths (not necessarily simple) from node u to node v of
length d. Then |Pt+1(u1, v1)| = |Pt+1(u2, v2)|.
• Denote Qd(u, v) be the set of all hitting paths (not necessarily simple) from node u to node
v of length d. Then, {{ω(Q) : Q ∈ Qt+1(u1, v1)}} = {{ω(Q) : Q ∈ Qt+1(u2, v2)}}, and
{{ω(Q) : Q ∈ Qt+1(v1, u1)}} = {{ω(Q) : Q ∈ Qt+1(v2, u2)}}.
Proof. We prove the lemma by induction over iteration t. We first prove the base case t = 0.
• If u1 = v1, then by Lemma C.52 u2 = v2. Note that obviously |P1(u, u)| = 0 and
|Q1(u, u)| = 0 for any node u, namely |P1(u1, u1)| = |P1(u2, u2)| and Q1(u1, u1) =
Q1(u2, u2) = ∅.
• Similarly, if u1 ̸= v1 and {u1, v1} /∈ E, then by Lemma C.52 u2 ̸= v2 and {u2, v2} /∈ E.
We also have |P1(u1, v1)| = |P1(u2, v2)| = 0 and Q1(u1, v1) = Q1(u2, v2) = ∅.
• If u1 ̸= v1 and {u1, v1} ∈ E, then by Lemma C.52 u2 ̸= v2 and {u2, v2} ∈ E. Then
|P1(u1, v1)| = |P1(u2, v2)| = 1 and Q1(u1, v1) = Q1(u2, v2) where both sets have a single
element that is an empty tuple (0-dimension).
Now suppose that the conclusion of Lemma C.53 holds in iteration t, we will prove that it also holds
in iteration t + 1. First note that for any two nodes u, v, |Pt+1(u, v)| = (cid:80)
w∈NG(v) |Pt+1(u, w)|. If
G (u1, v1) = χt+1
χt+1
G (u2, v2), then by definition of 2-FWL update formula (19)
{{(χt
G(u1, w), χt
G(w, v1)) : w ∈ V}} = {{(χt
G(u1, w) : w ∈ NG(v1)}} = {{χt
G(u2, w), χt
G(w, v2)) : w ∈ V}}.
G(u2, w) : w ∈ NG(v2)}} due to
which implies that {{χt
Lemma C.52. Therefore,
w∈NG(v1) |Pt+1(u1, w)| = (cid:80)
: w ∈ NG(v1)}} = {{|Pt+1(u2, w)|
: w ∈ NG(v2)}}.
w∈NG(v2) |Pt+1(u2, w)| and thus we have
• By induction, {{|Pt+1(u1, w)|
It follows that (cid:80)
|Pt+2(u1, v1)| = |Pt+2(u2, v2)|.
G(u1, w), χt
• By induction, {{(χt
G(u2, w), χt
{{(χt
Lemma C.52 says that χt
G(w, v1), {{ω(Q) : Q ∈ Qt+1(w, v1)}}) : w ∈ NG(u1)}} =
G(w, v2), {{ω(Q) : Q ∈ Qt+1(w, v2)}}) : w ∈ NG(u2)}}. Since
G(w, v) ̸= χt
G(v, v) if w ̸= v, we have
{{(χt
={{(χt
G(u1, w), {{ω(Q) : Q ∈ Qt+1(w, v1)}}) : w ∈ NG(u1)\{v1}}}
G(u2, w), {{ω(Q) : Q ∈ Qt+1(w, v2)}}) : w ∈ NG(u2)\{v2}}}
46
Published as a conference paper at ICLR 2023
Further using the third bullet of Lemma C.52 and rearranging the two multisets yields
{{(degG(w), ω(Q)) : w ∈ NG(u1)\{v1}, Q ∈ Qt+1(w, v1)}}
={{(degG(w), ω(Q)) : w ∈ NG(u2)\{v2}, Q ∈ Qt+1(w, v2)}}.
Equivalently, {{ω(Q) : Q ∈ Qt+2(u1, v1)}} = {{ω(Q) : Q ∈ Qt+2(u2, v2)}}. We can
similarly prove that {{ω(Q) : Q ∈ Qt+2(v1, u1)}} = {{ω(Q) : Q ∈ Qt+2(v2, u2)}}.
This concludes the proof of the induction step.
The above lemma directly yields the following corollary:
Corollary C.54. Given nodes u1, u2, v1, v2 ∈ V, if χG(u1, v1) = χG(u2, v2), then disG(u1, v1) =
disG(u2, v2) and disR
G(u1, v1) = disR
G(u2, v2).
G(u1, v1) = χt
Proof. If χG(u1, v1) = χG(u2, v2), then χt
G(u2, v2) holds for all t ≥ 0. By
Lemma C.53 |Pt(u1, v1)| = |Pt(u2, v2)| holds for all t ≥ 0 (the case t = 0 trivially holds).
Since disG(u, v) = min{t : |Pt(u1, v1)| > 0}, we conclude that disG(u1, v1) = disG(u2, v2). As
for the Resistance Distance disR
G, it is equivalent to the Commute Time Distance multiplied by a
G(u, w) = 2|E| disR
constant (Chandra et al., 1996, see also Appendix E.2), i.e. disC
G(u, w). Since
disC
P ∈Qi(v,u) q(P )) where Qi(u, v) is the set contain-
for a path
P ∈Qi(u2,v2) q(P ) and
P ∈Qi(v2,u2) q(P ) for all i ≥ 0 (the case i = 0 trivially holds) and thus
G(u2, v2).
ing all hitting paths of length i from u to v, and q(P ) = 1/
P = (x0, · · · , xd). By Lemma C.53, we have (cid:80)
(cid:80)
degG(u) (cid:81)d−1
P ∈Qi(u1,v1) q(P ) = (cid:80)
P ∈Qi(v1,u1) q(P ) = (cid:80)
G(u1, v1) = disC
P ∈Qi(u,v) q(P ) + (cid:80)
G(u2, v2), namely disR
G(u1, v1) = disR
G(u, v) = (cid:80)∞
i=0 i · ((cid:80)
i=1 deg(xi)
disC
(cid:16)
(cid:17)
We are now ready to prove Theorem 4.5.
Theorem C.55. The 2-FWL algorithm is more powerful than both SPD-WL and RD-WL. Formally,
given a graph G, let χ2FWL
be the vertex color mappings for these algo-
and χRDWL
rithms, respectively. Then the partition induced by χ2FWL
.
G
is finer than both χSPDWL
and χRDWL
G
, χSPDWL
G
G
G
G
Proof. Note that by definition (see Appendix B.2), we have χG(v) := χG(v, v) for any node v ∈ V.
If χG(v1) = χG(v2), then by definition of 2-FWL aggregation formula,
{{(χG(v1, w), χG(w, v1)) : w ∈ V}} = {{(χG(v2, w), χG(w, v2)) : w ∈ V}}.
Using Lemma C.6, if χG(v1, w1) = χG(v2, w2) for some nodes w1 and w2, then χG(w1) =
χG(w2). Therefore, by using Corollary C.54 we obtain that if χG(v1) = χG(v2), then
{{(χG(w), disG(w, v1)) : w ∈ V}} = {{(χG(w), disG(w, v2)) : w ∈ V}},
{{(χG(w), disR
G(w, v2)) : w ∈ V}}.
G(w, v1)) : w ∈ V}} = {{(χG(w), disR
The above equantions show that the partition induced by χ2FWL
χRDWL
and conclude the proof.
G
G
is finer than both χSPDWL
G
and
Finally, the following proposition trivially holds and will be used to prove Corollary 4.6.
Proposition C.56. Given a graph G = (V, EG), let χG and ˜χG be two color mappings induced by
two different (general) color refinement algorithms, respectively. If the vertex partition induced by
the mapping χG is finer than that of ˜χG, then:
• The mapping χG can distinguish cut vertices/edges if ˜χG can distinguish cut vertices/edges;
• The mapping χG can distinguish the isomorphism type of BCVTree(G)/BCETree(G) if ˜χG
can distinguish the isomorphism type of BCVTree(G)/BCETree(G).
Corollary 4.6 is a simple consequence of Theorem 4.5 and Proposition C.56.
47
Published as a conference paper at ICLR 2023
Dodecahedron
Desargues graph
(a) SPD-WL fails while RD-WL succeeds.
4x4 rook’s graph
Shrikhande graph
(b) Both SPD-WL and RD-WL fail.
Figure 12: Illustration of non-isomorphic distance-regular graphs.
C.7 REGARDING DISTANCE-REGULAR GRAPHS
In this subsection, we give more fine-grained theoretical results on the expressiveness upper bound
of GD-WL by considering the special problem of distinguishing distance-regular graphs, a class of
hard graphs that are highly relevant to the GD-WL framework. We provide a full characterization of
what types of distance-regular graphs different GD-WL algorithms can or cannot distinguish, with
both proofs and counterexamples.
Given a graph G = (V, E), let N i
G(u) = {w ∈ V : disG(u, w) = i} be the i-hop neighbors
of u in G and let D(G) := maxu,v∈V disG(u, v) be the diameter of G. We say G is distance-
regular if for all i, j ∈ [D(G)] and all nodes u, v, w, x ∈ V with disG(u, v) = disG(w, x), we have
|N i
G(x)|. From the definition, it is straightforward to see that for all
u, v ∈ V and i ∈ [D(G)], |N i
G(v)|, i.e., the number of i-hop neighbors is the same for all
nodes. We thus denote κ(G) = (k1, · · · , kD(G)) as the k-hop-neighbor array where ki := |N i
G(u)|
with u ∈ V chosen arbitrarily. We next define another important array:
G(v)| = |N i
G(u)| = |N i
G(w) ∩ N j
G(u) ∩ N j
Definition C.57. (Intersection array) The intersection array of a distance-regular graph G is
denoted as ι(G) = {b0, · · · , bD(G)−1; c1, · · · , cD(G)} where bi = |NG(u) ∩ N i+1
G (v)| and
ci = |NG(u) ∩ N i−1
G (v)| with disG(u, v) = i.
We now present our main results.
Theorem C.58. Let G and H be two connected distance-regular graphs. Then the following holds:
• SPD-WL can distinguish the two graphs if and only if their k-hop-neighbor arrays differ, i.e.
κ(G) ̸= κ(H).
• RD-WL can distinguish the two graphs if and only if their intersection arrays differ, i.e.
ι(G) ̸= ι(H).
• 2-FWL can distinguish the two graphs if and only if their intersection arrays differ, i.e.
ι(G) ̸= ι(H).
Theorem C.58 precisely characterizes the equivalence class of all distance-regular graphs for differ-
ent types of algorithms. Combined the fact that ι(G) = ι(H) implies κ(G) = κ(H) (see e.g. van
Dam et al. (2014, page 8)), we immediately arrive at the following corollary:
Corollary C.59. RD-WL is strictly more powerful than SPD-WL in distinguishing non-isomorphic
distance-regular graphs. Moreover, RD-WL is as powerful as 2-FWL in distinguishing non-
isomorphic distance-regular graphs.
Counterexamples. We provide representitive counterexamples in Figure 12 for both SPD-WL and
RD-WL. In Figure 12(a), both the Dodecahedron and the Desargues graph have 20 vertices and
the same k-hop-neighbor array (3, 6, 6, 3, 1), and thus SPD-WL cannot distinguish them. How-
ever, they have the different intersection array (i.e., {3, 2, 1, 1, 1; 1, 1, 1, 2, 3} for Dodecahedron and
{3, 2, 2, 1, 1; 1, 1, 2, 2, 3} for the Desargues graph), and thus RD-WL can distinguish them. In Fig-
ure 12(b), we make use of the well-known 4x4 rook’s graph and the Shrikhande graph, both of which
are strongly regular and thus distance-regular. They have the same intersection array {6, 3; 1, 2} and
thus both RD-WL and 2-FWL cannot distinguish them although they are non-isomorphic.
48
Published as a conference paper at ICLR 2023
C.7.1 PROOF OF THEOREM C.58
We first present a lemma that links the definition of distance-regular graph to its intersection array.
The proof is based on the Bose-Mesner algebra and its association scheme, and please refer to van
Dam et al. (2014, Sections 2.5 and 2.6) for details.
Lemma C.60. Let G and H be two graphs with the same intersection array, and suppose nodes
u, v, w, x satisfy disG(u, v) = disG(w, x). Then |N i
H (x)| for all
i, j ∈ N.
G(v)| = |N i
H (w) ∩ N j
G(u) ∩ N j
G(u)| = |N i
G of graph G after the first iteration. Then for two graphs G, H with n nodes, χ1
Proof of the first item of Theorem C.58. This part is straightforward. Consider the SPD-WL color
mapping χ1
G(u) =
χ1
H (v) if and only if |N i
H (v)| for all i ∈ [n − 1]. Therefore, if κ(G) ̸= κ(H), then for
any node u in G and v in H, |N j
H (v)| holds for some j ∈ [max(D(G), D(H))] and
thus χ1
H (v). Namely, χG(u) ̸= χH (v) for all nodes u in G and v in H, implying that
SPD-WL can distinguish the two graphs. On the other hand, if κ(G) = κ(H), then for any node u
H (v) for any iteration t ∈ N, and
in G and v in H we have χ1
thus SPD-WL cannot distinguish the two graphs.
H (v). Similarly, χt
G(u)| = |N j
G(u) ̸= χ1
G(u) = χ1
G(u) = χt
Proof of the second item of Theorem C.58. The key insight is that given a distance-regular graph,
the resistance distance between a pair of nodes (u, v) only depends on its SPD. Formally, for any
nodes u, v, w, x in a distance-regular graph G, disG(u, v) = disG(w, x) implies that disR
G(u, v) =
disR
G(w, x). Actually, we have the following stronger result:
Theorem C.61. For any two nodes u, v in a connected distance-regular graph G, disR
rdisG(u,v) where the sequence {rd}D(G)
d=0 is recursively defined as follows:
G(u, v) =
rd =



0
rd−1 +
2
nkd−1bd−1
D(G)
(cid:88)
i=d
if d = 0,
ki
if d ∈ [D(G)],
(20)
where ι(G) = {b0, · · · , bD(G)−1; c1, · · · , cD(G)} is the intersection array of G and κ(G) =
(k1, · · · , kD(G)) is its k-hop-neighbor array.
Proof. Let R ∈ Rn×n be the RD matrix. Based on Theorem E.1, R can be expressed as R =
diag(M)11⊤ + 11⊤ diag(M) − 2M, where M = (cid:0)L + 1
and L is the graph Laplacian
matrix. Now let (cid:101)R = [rdisG(u,v)]u,v∈V be the matrix with elements defined in (20). The key step is
to prove that 2M = c11⊤ − (cid:101)R for some c ∈ R. This will yield
n 11⊤(cid:1)−1
R =
1
2
(cid:16)
(cid:17)
diag(c11⊤ − (cid:101)R)11⊤ + 11⊤ diag(c11⊤ − (cid:101)R)
− c11⊤ + (cid:101)R = (cid:101)R
(since diag( (cid:101)R) = O) and finish the proof.
We now prove 2M = c11⊤ − (cid:101)R for some c ∈ R, namely (cid:0)L + 1
= 2I. Note
that (cid:101)R is a symmetric matrix and satisfy (cid:101)R1 = c11 for some c1 ∈ R because G is distance-regular.
Combined the fact that L1 = 0, we have
(cid:17)
c11⊤ − (cid:101)R
n 11⊤(cid:1) (cid:16)
(cid:18)
L +
(cid:19) (cid:16)
1
n
11⊤
c11⊤ − (cid:101)R
(cid:17)
(cid:16)
c −
=
(cid:17)
c1
n
11⊤ − L (cid:101)R.
It thus suffices to prove that L (cid:101)R = c11⊤ − 2I for some c ∈ R. Let us calculate each element
[L (cid:101)R]uv (u, v ∈ V). We have
[L (cid:101)R]uv = k1rdisG(u,v) −
D(G)
(cid:88)
d=0
rd|NG(u) ∩ N d
G(v)|.
(21)
Now consider the following three cases:
49
Published as a conference paper at ICLR 2023
• u = v. In this case, rdisG(u,v) = 0 and we have
D(G)
(cid:88)
d=1
rd|NG(u) ∩ N d
G(v)| = r1k1 =
2(n − 1)
n
by using b0 = k1 and k0 = 0. Thus [L (cid:101)R]uv = − 2(n−1)
n
.
• u ̸= v and disG(u, v) < D(G). Denote j = disG(u, v). In this case, in (21) the term
G(v) ̸= ∅ only when d ∈ {j − 1, j, j + 1}, and by definition of intersection array
G(v)| =
NG(u) ∩ N d
we have |NG(u) ∩ N j−1
|NG(u)| − cj − bj = k1 − cj − bj. Therefore,
G (v)| = bj, and |NG(u) ∩ N j
G (v)| = cj, |NG(u) ∩ N j+1
[L (cid:101)R]uv = k1rj − rj−1cj − rj(k1 − bj − cj) − rj+1bj
= cj(rj − rj−1) + bj(rj − rj+1)
D(G)
(cid:88)
D(G)
(cid:88)
=
2cj
nkj−1bj−1

D(G)
(cid:88)
=
2
nkj

i=j
kj −
2bj
nkjbj

ki −
D(G)
(cid:88)
kj
 =
2
n
,
i=j
i=j+1
ki
i=j+1
where in the second last step we use the recursive relation of rj, and in the last step we use
the fact that kj = kj−1bj−1
for any j ∈ [D(G)] (see e.g. van Dam et al. (2014, page 8)).
cj
• u ̸= v and disG(u, v) = D(G). This case is similar as the previous one. Denote j =
disG(u, v), and NG(u) ∩ N d
G(v) ̸= ∅ only when d ∈ {j − 1, j}. We have
[L (cid:101)R]uv = k1rj − rj−1cj − rj(k1 − cj)
= cj(rj − rj−1)
2cj
nkj−1bj−1
=
kj =
2
n
,
where we again use kj = kj−1bj−1
cj
.
Combining the above three cases, we conclude that L (cid:101)R = 2
n 11⊤ −2I, which finishes the proof.
We are now ready to prove the main result. Let G = (VG, EG) and H = (VH , EH ) be two distance-
regular graphs. We first prove that if ι(G) = ι(H), then RD-WL cannot distinguish the two graphs.
This is a simple consequence of Theorem C.61. Combined with the fact that κ(G) = κ(H), we
have {disR
H (v, w) : w ∈ VH } for any nodes u ∈ VG and v ∈ VH .
G and χ1
Therefore, after the first iteration, the RD-WL color mappings χ1
H (v)
for all u ∈ VG and v ∈ VH . Similarly, after the t-th iteration we still have χt
H (v) for all
u ∈ VG and v ∈ VH and thus RD-WL cannot distinguish the two graphs.
G(u, w) : w ∈ VG} = {disR
H satisfy χ1
G(u) = χ1
G(u) = χt
It remains to prove that if ι(G) ̸= ι(H), then RD-WL can distinguish the two graphs. First observe
that in Theorem C.61, ri < rj holds for any i < j. Therefore, for any nodes u ∈ VG and v ∈ VH ,
{disR
G(u, w) : w ∈ VG} = {disR
H (v, w) : w ∈ VH } if and only if
{{{ri(G)}} × ki(G) : i ∈ [D(G)]} = {{{ri(H)}} × ki(H) : i ∈ [D(H)]},
(22)
where {{r}} × k is a multiset containing k repeated elements of value r. If ι(G) ̸= ι(H), then there
exists a minimal index d such that bi(G) = bi(H) and ci+1(G) = ci+1(H) for all i < d but bi(G) ̸=
bi(H) or ci+1(G) ̸= ci+1(H). It follows by Theorem C.61 that ri(G) = ri(H) and ki(G) = ki(H)
for all i ≤ d but either rd+1(G) ̸= rd+1(H) (if bd(G) ̸= bd(H)) or kd+1(G) ̸= kd+1(H) (if
bd(G) = bd(H) and cd+1(G) ̸= cd+1(H)). Therefore, (22) does not hold and χ1
H (v) for
any u ∈ VG and v ∈ VH , namely, RD-WL can distinguish the two graphs.
G(u) ̸= χ1
Proof of the third item of Theorem C.58. First, if ι(G) ̸= ι(H), then 2-FWL can distinguish graphs
G and H. This is simply due to the fact that 2-FWL is more powerful than RD-WL (Theorem 4.5).
It remains to prove that if ι(G) = ι(H), then 2-FWL cannot distinguish graphs G and H.
50
Published as a conference paper at ICLR 2023
Let χt
G : VG × VG → C be the 2-FWL color mapping of graph G after t iterations. We aim to
prove that for any nodes u, v ∈ VG and w, x ∈ VH , if disG(u, v) = disG(w, x), then χt
G(u, v) =
H (w, x) for any t ∈ N. We prove it by induction. The base case of t = 0 trivially holds. Now
χt
suppose the case of t holds and let us consider the color mapping after t + 1 iterations. By the
2-FWL update rule (2),
G (u, v) = hash (cid:0)χt
χt+1
G(u, v), {{(χt
G(u, z), χt
G(z, v)) : z ∈ VG}}(cid:1) .
It thus suffices to prove that
G(u, z), χt
{{(χt
G(z, v)) : z ∈ VG}} = {{(χt
H (w, z), χt
H (z, x)) : z ∈ VH }}.
(23)
(24)
By Lemma C.60, we have
{{(disG(u, z), disG(z, v)) : z ∈ VG}} = {{(disH (w, z), disH (z, x)) : z ∈ VH }}.
This already yields (24) by the induction result of iteration t. We thus complete the proof.
D FURTHER DISCUSSIONS WITH PRIOR WORKS
D.1 KNOWN METRICS FOR MEASURING THE EXPRESSIVE POWER OF GNNS
In this subsection, we review existing metrics used in prior works to measure the expressiveness of
GNNs. We will discuss the limitations of these metrics and argue why biconnectivity may serve as
a more reasonable and compelling criterion in designing powerful GNN architectures.
WL hierarchy. Since the discovery of the relationship between MPNNs and 1-WL test (Xu et al.,
2019; Morris et al., 2019), the WL hierarchy has been considered as the most standard metric to
guide designing expressive GNNs. However, achieving an expressive power that matches the 2-
FWL test is already highly difficult. Indeed, each iteration of the 2-FWL algorithm already requires
a complexity of Ω(n3) time and Θ(n2) space for a graph with n vertices (Immerman & Lander,
1990). Therefore, it is impossible to design expressive GNNs using this metric while maintain-
ing its computational efficiency. Moreover, whether achieving higher-order WL expressiveness is
necessary and helpful for real-world tasks has been questioned by recent works (Veliˇckovi´c, 2022).
Structural metrics. Another line of works thus sought different metrics to measure the expressive
power of GNNs. Several popular choices are the ability of counting substructures (Arvind et al.,
2020; Chen et al., 2020; Bouritsas et al., 2022), detecting cycles (Loukas, 2020; Vignac et al., 2020;
Huang et al., 2023), calculating the graph diameter (Garg et al., 2020; Loukas, 2020) or other graph-
related (combinatorial) problems (Sato et al., 2019). Yet, all these metrics have a common drawback:
the corresponding problems may be too hard for GNNs to solve.
Indeed, we show in Table 4
that solving any above task requires a computation complexity that grows super-linear w.r.t.
the
graph size even using advanced algorithms. Therefore, it is quite natural that standard MPNNs
are not expressive for these metrics, since no GNNs can solve these tasks while being efficient.
Consequently, instead of using GNNs to directly learn these metrics, these works had to use a
precomputation step which can be costly in the worst case.
Table 4: The best computational complexity of known algorithms for solving different graph prob-
lems. Here n and m are the number of nodes and edges of a given graph, respectively.
Metric
Complexity
Reference
k-FWL
Counting/detecting triangles
Detecting cycles of an odd length k ≥ 3
Detecting cycles of an even length k ≥ 4 O(n2)
O(nm)
Calculating the graph diameter
Ω(nk+1)
O(min(n2.376, m3/2))
O(min(n2.376, m2))
(Immerman & Lander, 1990)
(Alon et al., 1997)
(Alon et al., 1997)
(Yuster & Zwick, 1997)
–
Detecting cut vertices
Detecting cut edges
Θ(n + m)
Θ(n + m)
(Tarjan, 1972)
(Tarjan, 1972)
Due to the lack of proper metrics, most subsequent works mainly justify the expressive power of
their proposed GNNs by focusing on regular graphs (Li et al., 2020; Bevilacqua et al., 2022; Bodnar
51
Published as a conference paper at ICLR 2023
et al., 2021b; Feng et al., 2022; Velingker et al., 2022, to list a few), which hardly appear in practice.
In contrast, the biconnectivity metrics proposed in this paper are different from all prior metrics, in
that (i) it is a basic graph property and has significant values in both theory and applications; (i) it
can be efficiently calculated with a complexity linear in the graph size, and thus it is reasonable to
expect that these metrics should be learned by expressive GNNs.
D.2 GNNS WITH DISTANCE ENCODING
In this subsection, we review prior works that are related to our proposed GD-WL. In the research
field of expressive GNNs, the idea of incorporating distance first appeared in Li et al. (2020), where
the authors mainly considered using distance encoding as node features and showed that distance
can help distinguish regular graphs. They also considered an approach similar to k-hop aggrega-
tion by incorporating distance into the message-passing procedure (but without a systematic study).
Zhang & Li (2021) designed a subgraph GNN that also uses (generalized) distance encoding as
node features in each subgraph. Ying et al. (2021a) designed a Transformer architecture that incor-
porates distance information and empirically showed excellent performance. Very recently, Feng
et al. (2022) formally studied the expressive power of k-hop GNNs. Yet, they still restricted the
analysis to regular graphs. The concurrent work of Abboud et al. (2022) designed the shortest path
network which is highly similar to our proposed SPD-WL. They showed the resulting model can
alleviate the bottlenecks and over-squashing problems for MPNNs (Alon & Yahav, 2021; Topping
et al., 2022) due to the increased receptive field.
Compared with prior works, our contribution lies in the following three aspects:
• We formalize the principled and more expressive GD-WL framework, which comprises
SPD-WL as a special case. Our framework is theoretically clean and generalizes all prior
works in a unified manner.
• We systematically and theoretically analyze the expressive power of SPD-WL for general
graphs and highlight a fundamental advantage in distinguishing edge-biconnectivity.
• We design a Transformer-based GNN that is provably as expressive as GD-WL. Thus, our
framework is not only for theoretical analysis, but can also be easily implemented with good
empirical performance on real-world tasks.
Discussions with the concurrent work of Velingker et al. (2022). After the initial submission, we
became aware of a concurrent work (Velingker et al., 2022) which also explored the use of Resistance
Distance to enhance the expressiveness of standard MPNNs. Here, we provide a comprehensive
comparison of these two works. Overall, the main difference is that their approach incorporates
RD (and several related affinity measures) into node/edge features (like Zhang & Li (2021)), while
we combine RD to design a new WL aggregation procedure. As for the theoretical analysis, they
only give a few toy examples of regular graphs to justify the expressive power beyond the 1-WL
test, while we give a systematic analysis of the power of RD-WL for general graphs and point out
that it is fully expressive for vertex-biconnectivity. In Velingker et al. (2022), the authors also made
comparisons to SPD and conjectured that RD may have additional advantages than SPD in terms of
expressiveness. In fact, this question is formally answered in our work, by proving that RD-WL is
expressive for vertex-biconnectivity while SPD-WL is not. Another important contribution of our
work is that we provide an upper bound of the expressive power of RD-WL to be 2-FWL (3-WL),
which reveals the limit of incorporating RD information. We also provide a precise and complete
characterization for the expressiveness of RD-WL in distinguishing distance-regular graphs, which
reveals that RD-WL can match the power of 2-FWL in distinguishing these hard graphs.
E IMPLEMENTATION OF GENERALIZED DISTANCE WEISFEILER-LEHMAN
In this section, we give implementation details of GD-WL and our proposed GNN architecture. We
also give detailed analysis of its computation complexity. Below, assume the input graph G = (V, E)
has n vertices and m edges.
52
Published as a conference paper at ICLR 2023
E.1 PREPROCESSING SHORTEST PATH DISTANCE
Shortest Path Distance can be easily calculated using the Floyd-Warshall algorithm (Floyd, 1962),
which has a complexity of Θ(n3). For sparse graphs typically encountered in practice (i.e. m =
o(n2)), a more clever way is to use breadth-first search that computes the distance from a given node
to all other nodes in the graph. The time complexity can be improved to Θ(nm).
E.2 PREPROCESSING RESISTANCE DISTANCE
In this subsection, we first describe several important properties of Resistance Distance. Based on
these properties, we give a simple yet efficient algorithm to calculate Resistance Distance.
G(u, v) = 2m disR
Equivalence between Resistance Distance (RD) and Commute Time Distance (CTD). Chan-
dra et al. (1996) established an important relationship between RD and CTD, by proving that
disC
G(u, v) holds for any graph G and any nodes u, v ∈ V. Here, the Commute
Time Distance is defined as disC
G(u, v) := hG(u, v)+hG(v, u) where hG(u, v) is the average hitting
time from u to v in a random walk. Concretely, hG(u, v) is equal to the average number of edges
passed in a random walk when starting from u and reaching v for the first time. Mathmatically, it
satisfies the following recursive relation:
hG(u, v) =



0
∞
1 +
1
degG(u)
(cid:80)
w∈NG(u) hG(u, v) otherwise.
if u = v,
if u and v are in different connected components,
(25)
The above equation can be used to calculate CTD and thus RD, as we will show later.
Resistance Distance is a graph metric. We say a function dG : V × V → R is a graph metric if
it is non-negative, positive semidefinite, symmetric, and satisfies triangular inequality. Let G be a
connected graph. Then Resistance Distance disR
G is a valid graph metric because:
G(u, v) ≥ 0 holds for any u, v ∈ V. Moreover, disR
G(u, v) =
• (Positive semidefiniteness) disR
0 iff u = v.
G(v, u) holds for any u, v ∈ V.
G(u, v) = disR
• (Symmetry) disR
G(v, w) ≥ disR
• (Triangular Inequality) For any u, v, w ∈ V, disR
G(u, w). This
can be seen from the definition of CTD, since disC
G(v, w) is equal to the average
hitting time from u to w under the condition of passing node v, which is obviously larger
than disR
G(u, v) + disR
G(u, v)+disC
G(u, w).
Comparing RD with SPD. It is easy to see that RD is always no larger than SPD, i.e. disR
G(u, v) ≤
disG(u, v). This is because for any subgraph G′ of G, we have disR
G′(u, v), and when
G′ is chosen to contain only the edges that belong to the shortest path between u and v, we have
disR
G(u, v) ≤
n − 1. However, unlike SPD which is an integer, RD can be a general rational number. RD can thus
be seen as a more fine-grained distance metric than SPD. Nevertheless, RD is still discrete and there
are only finitely many possible values of disR
G′(u, v) = disG(u, v). Therefore, the range of RD is the same as SPD, i.e. 0 ≤ disR
G(u, v) when n is fixed.
Relationship to graph Laplacian. We have the following theorem:
Theorem E.1. Let G = (V, E) be a connected graph, V = [n], and let L ∈ Sn be the graph
Laplacian. Then
G(u, v) ≤ disR
where M ∈ Sn is a symmetric matrix defined as
disR
G(i, j) = Mi,i + Mj,j − 2Mi,j,
(cid:18)
M =
L +
(cid:19)−1
.
1
n
11⊤
Proof. Denote d = (degG(1), · · · , degG(n))⊤. Define the probability matrix P such that Pij = 0
if {i, j} /∈ E and Pij = 1/ degG(i) if {i, j} ∈ E. Then for any i ̸= j, (25) can be equivalently
53
Published as a conference paper at ICLR 2023
written as
h(i, j) = 1 +
n
(cid:88)
k=1
Pikh(k, j) − Pijh(j, j).
(26)
Now define a matrix ˜H such that ˜Hij = 1 + (cid:80)n
i ̸= j (although ˜Hii ̸= 0 = h(i, i)). ˜H can be equivalently written as
k=1 Pik ˜Hkj − Pij ˜Hjj, then ˜Hij = h(i, j) for all
˜H = 11⊤ + P ˜H − P diag( ˜H),
where diag( ˜H) is the diagnal matrix with elements ˜Hii for i ∈ [n].
We first calculate diag( ˜H). Noting that d⊤P = d, we have
d⊤ ˜H = d⊤11⊤ + d⊤( ˜H − diag( ˜H)),
and thus d⊤ diag( ˜H) = d⊤11⊤, namely
˜Hii =
1
di
d⊤1 =
2m
di
.
(27)
(28)
Now define H = ˜H − diag( ˜H), then Hij = h(i, j) for all i, j ∈ [n]. We will calculate H in
the following proof. We first write (27) equivalently as H + diag( ˜H) = 11⊤ + PH. Then by
multiplying D, we have
D(I − P)H = D11⊤ − D diag( ˜H).
Using the fact that D(I − P) = L and (28), we obtain
LH = D11⊤ − 2mI.
Next, noting that L1 = 0, we have
(cid:18)
L =
L +
1
n
11⊤
(cid:19) (cid:18)
I −
(cid:19)
.
11⊤
1
n
(29)
(30)
(31)
One important property is that the matrix (cid:0)L + 1
Theorem 4) for a proof). Combining (30) and (31) we have
n 11⊤(cid:1) is invertible (see Gutman & Xiao (2004,
(cid:18)
I −
(cid:19)
11⊤
1
n
(cid:18)
H =
L +
(cid:19)−1
1
n
11⊤
(cid:0)D11⊤ − 2mI(cid:1) = M (cid:0)D11⊤ − 2mI(cid:1) .
(32)
By taking diagonal elements and noting that diag(H) = O, we otain
−
1
n
diag (cid:0)11⊤H(cid:1) = diag (cid:0)MD11⊤(cid:1) − 2m diag (M)
Namely,
1
n
Substituting (34) into (32) yields
H⊤1 = −MD1 + 2m diag (M) 1.
H = M (cid:0)D11⊤ − 2mI(cid:1) − 11⊤DM + 2m11⊤ diag (M) .
(33)
(34)
(35)
Therefore,
This finally yields disR
the proof.
H + H⊤ = 2m(11⊤ diag (M) + diag (M) 11⊤ − 2M).
G(i, j) = 1
(36)
2m (H+H⊤) = Mi,i+Mj,j −2Mi,j and concludes
G(i, j) = 1
2m disC
Computational Complexity. The graph Laplacian can be calculated in O(n2) time, and M can
be calculated by matrix inversion which requires O(n3) time. Therefore, the overall computational
complexity is O(n3) (or O(n2.376) using advanced matrix multiplication algorithms).
For sparse graphs typically encountered in practice (i.e. m = o(n2)), one may similarly ask whether
a complexity that depends on m can be achieved. We conjecture that it should be possible. Below,
54
Published as a conference paper at ICLR 2023
we give another algorithm to calculate (cid:0)L + 1
equivalently written as L = EE⊤, where E ∈ Rn×m is defined as
n 11⊤(cid:1)−1
. Note that the graph Laplacian L can be
(cid:40) 1
Eij =
if ϵj = {i, k} and k > i
if ϵj = {i, k} and k < i
if i /∈ ϵj
where we denote E = {ϵ1, · · · , ϵm}. Let E = [e1, · · · , em] where ei ∈ Rn, then M =
(cid:0) 1
n 11⊤ + (cid:80)m
. Noting that each ei is highly sparse with only two non-zero elements.
We suspect that one can obtain an O(nm) complexity using techniques similar to the Sherman-
Morrison-Woodbury update. We leave it as an open problem.
i=1 eie⊤
i
−1
0
(37)
(cid:1)−1
E.3 TRANSFORMER-BASED IMPLEMENTATION
Graphormer-GD. The model is built on the Graphormer (Ying et al., 2021a) model, which use the
Transformer (Vaswani et al., 2017) as the backbone network. A Transformer block consists of two
layers: a self-attention layer followed by a feed-forward layer, with both layers having normalization
(e.g., LayerNorm (Ba et al., 2016)) and skip connections (He et al., 2016). Denote X(l) ∈ Rn×d as
the input to the (l + 1)-th block and define X(0) = X, where n is the number of nodes and d is the
feature dimension. For an input X(l), the (l + 1)-th block works as follows:
K )⊤(cid:17)
Ah(X(l)) = softmax
Q (X(l)Wl,h
X(l)Wl,h
(38)
(cid:16)
;
ˆX(l) = X(l) +
H
(cid:88)
h=1
Ah(X(l))X(l)Wl,h
V Wl,h
O ;
X(l+1) = ˆX(l) + GELU( ˆX(l)Wl
1)Wl
2,
1 ∈ Rd×r, Wl
(39)
(40)
O ∈ RdH ×d, Wl,h
where Wl,h
2 ∈ Rr×d, H is the number
V ∈ Rd×dH , Wl
of attention heads, dH is the dimension of each head, and r is the dimension of the hidden layer.
Ah(X) is usually referred to as the attention matrix.
Q , Wl,h
K , Wl,h
Note that the self-attention layer and the feed-forward layer introduced in (39) and (40) do not
encode any structural information of the input graph. As stated in Section 4, we incorporate the
distance information into the attention layers of our Graphormer-GD model. The calculation of the
attention matrix in (38) is modified as:
Ah(X(l)) = ϕl,h
1 (D) ⊙ softmax
(cid:16)
X(l)Wl,h
Q (X(l)Wl,h
K )⊤ + ϕl,h
(cid:17)
2 (D)
;
(41)
where D ∈ Rn×n is the distance matrix such that Duv = dG(u, v), ϕh
functions applied to D, and ⊙ denotes the element-wise multiplication.
structural information can be captured by our Graphormer-GD model.
1 and ϕh
2 are element-wise
In this way, the graph
As stated in Section 4, we mainly consider two distance metrics: Shortest Path Distance disG and
Resistance Distance disR
G. For SPD, we follow Ying et al. (2021a) to use their shortest path distance
encoding. Formally, let DSPD be the SPD matrix such that DSPD
uv = disG(u, v). The function ϕ1
and ϕ2 can simply be parameterized by two learnable vectors v1 and v2, so that ϕ1(DSPD
uv ) is a
learnable scalar corresponding to v1
(and similarly for ϕ2). If two nodes u and v are not in
the same connected component, i.e., DSPD
uv = ∞, a special learnable scalar is assigned. For RD,
we use the Gaussian Basis kernels (Scholkopf et al., 1997) to encode the value since it may not be
an integer. The encoded values from different Gaussian Basis kernels are concatenated and further
transformed by a two-layer MLP. We integrate both the SPD encoding and the RD encoding to
obtain ϕl,h
2 (D). Note that these two matrices are parameterized by different sets of
parameters. Following Ying et al. (2021a), we also incorporate the degree of each node in the input
layer using a degree embedding.
1 (D) and ϕl,h
DSPD
uv
Relationship between Graphormer-GD and GD-WL. As stated in Section 4, the expressive power
of Graphormer-GD is at most as powerful as GD-WL. We will prove that it is actually as powerful as
GD-WL under mild assumptions. We first restate the Lemma 5 from Xu et al. (2019), which shows
that sum aggregators can represent injective functions over multisets.
55
Published as a conference paper at ICLR 2023
Lemma E.2. (Xu et al., 2019, Lemma 5) Assume the set X is countable. Then there exists a function
f : X → Rn so that the function h( ˆX ) := (cid:80)
x∈ ˆX f (x) is unique for each multiset ˆX ⊂ X of
bounded size. Moreover, any multiset function g can be decomposed as g( ˆX ) = ϕ((cid:80)
x∈ ˆX f (x)) for
some function ϕ.
We are now ready to present the detailed proof of the Theorem 4.4, which is restated as follows:
Theorem E.3. Graphormer-GD is at most as powerful as GD-WL. Moreover, when choosing proper
functions ϕh
2 and using a sufficiently large number of heads and layers, Graphormer-GD is
as powerful as GD-WL.
1 and ϕh
Proof. Consider all graphs with no more than n nodes. The total number of possible values of both
SPD and RD are thus finite and depends on n (see Appendix E.2). Let
Dn = {(disG(u, v), disR
denote the set of all possible pairs (disG(u, v), disR
G(u, v)). Since Dn is finite, we can list
its elements as Dn = {dG,1, · · · , dG,|Dn|}. Without abuse of notation, denote dG(u, v) =
(disG(u, v), disR
G(u, v)). Then the GD-WL aggregation in (3) can be reformulated as follows:
G(u, v)) : G = (V, E), |V| ≤ n, u, v ∈ V}
(cid:16)
χt
G(v) := hash
G (v) := {{χt−1
where χt,k
(χt,1
G (v), χt,2
G (v), · · · , χt,|Dn|
G
(v))
(cid:17)
,
(42)
G (u) : u ∈ V, dG(u, v) = dG,k}}.
Intuitively, this reformulation indicates that in each iteration, GD-WL updates the color of node v
by hashing a tuple of color multisets, where each multiset is obtained by injectively aggregating
the colors of all nodes u ∈ V with certain distance configuration to node v. Therefore, to express
GD-WL, the model suffices to update the representation of each node following the above procedure.
K)⊤ + ϕh
1 (D) ⊙ softmax (cid:0)XWh
We show Graphormer-GD can achieve this goal. Recall that for the h-th head, the attention ma-
Q(XWh
trix is defined as ϕh
1 , we de-
fine it to be the indicator function ϕh
2 , we set it to
be constant irrespective to the matrix D. Let Wh
It can be seen that
the term softmax (cid:0)XWh
|V| 11⊤, and thus for each node v, the
output in the h-th attention head is the sum aggregation of representations of node u satisfying
dG(u, v) = dG,h. Formally,
Ah(X(l))X(l)(cid:105)
(cid:104)
1 (d) := I(d = dG,h). For the function ϕh
2 (D)(cid:1). For the function ϕh
2 (D)(cid:1) reduces to 1
K be zero matrices.
K)⊤ + ϕh
Q(XWh
X(l)(cid:105)
Q, Wh
(cid:88)
=
(cid:104)
.
1
|V|
v
dG(u,v)=dG,h
u
Note that the constant 1
|V| can be extracted with an additional head and be concatenated to the node
representations. Moreover, the node representation X is processed via the feed-forward network
in the previous layer (see (40). Thus, we can invoke Lemma E.2 and prove that the h-th atten-
tion head in Graphormer-GD can implement an injective aggregating function for {{χt−1
G (u) : u ∈
V, dG(u, v) = dG,h}}. Therefore, by using a sufficiently large number of attention heads, the multi-
set representations χt,k
G , k ∈ [|Dn|] can be injectively obtained.
Finally, the multi-head attention defined in (39) is equivalent to first concatenating the output of each
attention head and then using a linear mapping to transform the results. Thus, the concatenation
is clearly an injective mapping of the tuple of multisets
. When the linear
mapping has irrelational weights, the projection will also be injective. Therefore, one attention
layer followed by the feed-forward network can implement the aggregation formula (42). Thus, our
Graphormer-GD is able to simulate the GD-WL when using a sufficiant number of layers, which
concludes the proof.
G , ..., χt,|Dn|
G , χt,2
χt,1
(cid:16)
(cid:17)
G
F EXPERIMENTAL DETAILS
F.1 SYNTHETIC TASKS
Data Generation and Evaluation Metrics. We carefully design several graph generators to exam-
ine the expressive power of compared models on graph biconnectivity tasks. First, we include the
56
Published as a conference paper at ICLR 2023
two families of graphs presented in Examples C.9 and C.10 (Appendix C.2). We further introduce
a rich family of regular graphs with both cut vertices and cut edges. Each graph in this family is
constructed by first randomly generating several connected components and then linking them via
cut edges while simultaneously ensuring that each node has the same degree. Combining the above
three families of hard graphs, we online generate data instances to train the compared models. For
each data instance, the total number of nodes is upper bounded by 120. We use graph-level accu-
racy as the metric. That is, for each graph, the prediction of the model is considered correct only
when all and only the cut vertices/edges are correctly identified. We use different seeds to repeat the
experiments 5 times and report the average accuracy.
Baselines. We choose several baselines with their expressive power being at different levels. First,
we consider classic MPNNs including GCN (Kipf & Welling, 2017), GAT (Veliˇckovi´c et al., 2018),
and GIN (Bouritsas et al., 2022). The expressive power of these GNNs is proven to be at most
as powerful as the 1-WL test (Xu et al., 2019). We also compare the Graph Substructure Net-
work (Bouritsas et al., 2022), which extracts graph substructures to improve the expressive power of
MPNNs. The substructure counts are incorporated into node features or the aggregation procedure.
Lastly, we also compare the Graphormer model (Ying et al., 2021a), which achieved impressive
performance in several world competitions (Ying et al., 2021b; Shi et al., 2022; Luo et al., 2022a).
Settings. We employ a 6-layer Graphormer-GD model. The dimension of hidden layers and feed-
forward layers is set to 768. The number of Gaussian Basis kernels is set to 128. The number of
attention heads is set to 64. The batch size is set to 32. We use AdamW (Kingma & Ba, 2014) as the
optimizer and set its hyperparameter ϵ to 1e-8 and (β1, β2) to (0.9, 0.999). The peak learning rate is
set to 9e-5. The model is trained for 100k steps with a 6K-step warm-up stage. After the warm-up
stage, the learning rate decays linearly to 0. All models are trained on 1 NVIDIA Tesla V100 GPU.
F.2 REAL-WORLD TASKS
We conduct experiments on the popular benchmark dataset: ZINC from Benchmarking-GNNs
(Dwivedi et al., 2020). It is a real-world dataset that consists of 250K molecular graphs. The task
is to predict the constrained solubility of a molecule, which is an important chemical property for
drug discovery. We train our models on both the ZINC-Full and ZINC-Subset (12K selected graphs
following Dwivedi et al. (2020)).
Baselines. For a fair comparison, we set the parameter budget of the model to be around 500K fol-
lowing Dwivedi et al. (2020). We compare our Graphormer-GD with several competitive baselines,
which mainly fall into five categories: Message Passing Neural Networks (MPNNs), High-order
GNNs, Substructure-based GNNs, Subgraph GNNs, and Graph Transformers.
First, we compare several classic MPNNs including Graph Convolution Network (GCN) (Kipf &
Welling, 2017), Graph Isomorphism Network (GIN) (Xu et al., 2019), Graph Attention Network
(GAT) (Veliˇckovi´c et al., 2018), GraphSAGE (Hamilton et al., 2017) and MPNN(sum) (Gilmer
et al., 2017). Besides, we also include several popularly used models. Mixture Model Network
(MoNet) (Monti et al., 2017) introduces a general architecture to learn on graphs and manifolds us-
ing the Bayesian Gaussian Mixture Model. Gated Graph ConvNet (GatedGCN) considers residual
connections, batch normalization, and edge gates to design an anisotropic variant of GCN. We com-
pare the GatedGCN with positional encodings. Principal Neighborhood Aggregation (PNA) (Corso
et al., 2020) combines multiple aggregators with degree-scalers.
Second, we compare two higher-order Graph Neural Networks: RingGNN (Chen et al., 2019) and
3WLGNN (Maron et al., 2019a) following Dwivedi et al. (2020). RingGNN extends the family
of order-2 Graph G-invariant Networks without going into higher order tensors and is able to dis-
tinguish between non-isomorphic regular graphs where order-2 G-invariant networks provably fail.
3WLGNN uses rank-2 tensors to build the neural network and is proved to be equivalent to the 3-WL
test on graph isomorphism problems.
Third, we compare two representative types of substructure-based GNNs. The Graph Substructure
Network (Bouritsas et al., 2022) extracts graph substructures to improve the expressive power of
MPNNs. The substructure counts are incorporated into node features or the aggregation procedure.
We also compare the Cellular Isomorphism Network (Bodnar et al., 2021a), which extends theo-
57
Published as a conference paper at ICLR 2023
retical results on Simplicial Complexes to regular Cell Complexes. Such generalization provides a
powerful set of graph “lifting” transformations with a hierarchical message passing procedure.
Moreover, we compare several Subgraph GNNs. Nested Graph Neural Network (NGNN) (Zhang
& Li, 2021) represents a graph with rooted subgraphs instead of rooted subtrees. It extracts a local
subgraph around each node and applies a base GNN to each subgraph to learn a subgraph represen-
tation. The whole-graph representation is then obtained by pooling these subgraph representations.
GNN-AK (Zhao et al., 2022) follows a similar manner to develop Subgraph GNNs with different
generation policies. Equivariant Subgraph Aggregation Networks (ESAN) (Bevilacqua et al., 2022)
develops a unified framework that includes per-layer aggregation across subgraphs, which are gen-
erated using pre-defined policies like edge deletion and ego-networks. Subgraph Union Network
(SUN) (Frasca et al., 2022) is developed based on the symmetry analysis of a series of existing
Subgraph GNNs and an upper bound on their expressive power, which theoretically unifies previous
architectures and performs well across several graph representation learning benchmarks.
Last, we compare several Graph Transformer models. GraphTransformer (GT) (Dwivedi & Bres-
son, 2021) uses the Transformer model on graph tasks, which only aggregates the information from
neighbor nodes to ensure graph sparsity, and proposes to use Laplacian eigenvector as positional
encoding. Spectral Attention Network (SAN) (Kreuzer et al., 2021) uses a learned positional en-
coding (LPE) that can take advantage of the full Laplacian spectrum to learn the position of each
node in a given graph. Graphormer (Ying et al., 2021a) develops the centrality encoding, spatial
encoding, and edge encoding to incorporate the graph structure information into the Transformer
model. Universal RPE (URPE) (Luo et al., 2022b) first shows that there exist continuous sequence-
to-sequence functions which RPE-based Transformers cannot approximate, and develops a novel
and universal attention module called Universal RPE-based Attention. The effectiveness of URPE
has been verified across language and graph benchmarks (e.g., the ZINC dataset).
Settings. Our Graphormer-GD consists of 12 layers. The dimension of hidden layers and feed-
forward layers are set to 80. The number of Gaussian Basis kernels is set to 128. The number of
attention heads is set to 8. The batch size is selected from [128, 256, 512]. We use AdamW (Kingma
& Ba, 2014) as the optimizer, and set its hyperparameter ϵ to 1e-8 and (β1, β2) to (0.9, 0.999). The
peak learning rate is selected from [4e-4, 5e-4]. The model is trained for 600k and 800k steps with
a 60K-step warm-up stage for ZINC-Subset and ZINC-Full respectively. After the warm-up stage,
the learning rate decays linearly to zero. The dropout ratio is selected from [0.0, 0.1]. The weight
decay is selected from [0.0, 0.01]. All models are trained on 4 NVIDIA Tesla V100 GPUs.
F.3 MORE TASKS
Node-level Tasks. We further conduct experiments on real-world node-level tasks. Following Li
et al. (2020), we benchmark our model on two real-world graphs: Brazil-Airports and Europe-
Airports, both of which are air traffic networks and are collected by Ackland et al. (2005) from
the government websites. The nodes in each graph represent airports and each edge represents
that there are commercial flights between the connected nodes. The Brazil-Airports graph has 131
nodes, 1038 edges in total and its diameter is 5. The Europe-Airports graph has 399 nodes, 5995
edges in total and its diameter is 5. The airport nodes are divided into 4 different levels according
to the annual passenger flow distribution by 3 quantiles: 25%, 50%, and 75%. The task is to predict
the level of each airport node. We follow Li et al. (2020) to split the nodes of each graph into
train/validation/test subsets with the ratio being 0.8/0.1/0.1, respectively. The test accuracy of the
best checkpoint on the validation set is reported. We use different seeds to repeat the experiments
20 times and report the average accuracy.
Following Li et al. (2020), we choose several competitive baselines including classical MPNNs
(GCN, GraphSAGE, GIN), Struc2vec and Distance-encoding based GNNs (DE-GNN-SPD, DE-
GNN-LP, DEA-GNN-SPD). We refer interested readers to Li et al. (2020) for detailed descriptions
of baselines. For our Graphormer-GD, the dimension of hidden layers and feed-forward layers are
set to 80. The number of layers is selected from [3, 6]. The number of Gaussian Basis kernels is
set to 128. The number of attention heads is set to 8. The batch size is selected from [4, 8, 16, 32].
We use AdamW (Kingma & Ba, 2014) as the optimizer, and set its hyperparameter ϵ to 1e-8 and
(β1, β2) to (0.9, 0.999). The peak learning rate is selected from [2e-4, 7e-5, 4e-5]. The total number
of training steps is selected from [500, 1000, 2000]. The ratio of the warm-up stage is set to 10%.
58
Published as a conference paper at ICLR 2023
After the warm-up stage, the learning rate decays linearly to zero. The dropout ratio is selected from
[0.0, 0.1, 0.5]. All models are trained on 1 NVIDIA Tesla V100 GPUs.
The results are presented in Table 5. We can see that our model outperforms these baselines on both
datasets with a slightly larger variance value due to the small scale of the datasets.
Table 5: Average Accuracy on Brazil-Airports and Europe-Airports datasets. Experiments are re-
peated for 20 times with different seeds. We use \* to indicate the best performance.
Model
Brazil-Airports
Europe-Airports
GCN (Kipf & Welling, 2017)
GraphSAGE (Hamilton et al., 2017)
GIN (Xu et al., 2019)
Struc2vec (Ribeiro et al., 2017)
DE-GNN-SPD (Li et al., 2020)
DE-GNN-LP (Li et al., 2020)
DEA-GNN-SPD (Li et al., 2020)
64.55±4.18
70.65±5.33
71.89±3.60
70.88±4.26
73.28±2.47
75.10±3.80
75.37±3.25
54.83±2.69
56.29±3.21
57.05±4.08
57.94±4.01
56.98±2.79
58.41±3.20
57.99±2.39
Graphormer-GD (ours)
77.69±6.39\*
59.23±4.05\*
F.4 EFFICIENCY EVALUATION
We further conduct experiments to measure the efficiency of our approach by profiling the time
cost per training epoch. We compare the efficiency of Graphormer-GD with other baselines along
with the number of model parameters on the ZINC-subset from Dwivedi et al. (2020). The number
of layers and the hidden dimension of our Graphormer-GD are set to 12 and 80 respectively. The
number of attention heads is set to 8. The batch size is set to 128, which is the same as the settings of
all baselines. We run profiling of all models on a 16GB NVIDIA Tesla V100 GPU. For all baselines,
we evaluate the time costs based on the publicly available codes of Dwivedi et al. (2020) and Ying
et al. (2021a). The results are presented in Table 6.
From Table 6, we can draw the following conclusions. Firstly, the efficiency of Graphormer-GD is
in the same order of magnitude as classic MPNNs despite the fact that the computation complexity
of Graphormer-GD is higher than MPNNs (i.e., Θ(n2) v.s. Θ(n + m) for a graph with n nodes
and m edges). This may be due to the high parallelizability of the Transformer layers. Secondly,
Graphormer-GD is much more efficient than higher-order GNNs as reflected by the computation
complexity in Table 1. Finally, Graphormer-GD is almost as efficient as the original Graphormer,
since the newly introduced module to encode the Resistance Distance takes negligible additional
time compared to that of the whole architecture.
Table 6: Efficiency Evaluation of different GNN models. We report the time per training epoch
(seconds) as well as the number of model parameters.
Model
# Params
Time (s)
GCN (Kipf & Welling, 2017)
GraphSAGE (Hamilton et al., 2017)
MoNet (Monti et al., 2017)
GIN (Xu et al., 2019)
GAT (Veliˇckovi´c et al., 2018)
GatedGCN-PE (Bresson & Laurent, 2017)
RingGNN (Chen et al., 2019)
3WLGNN (Maron et al., 2019a)
Graphormer (Ying et al., 2021a)
Graphormer-GD (ours)
505,079
505,341
504,013
509,549
531,345
505,011
527,283
507,603
489,321
502,793
5.85
6.02
7.19
8.05
8.28
10.74
178.03
179.35
12.26
12.52
59
