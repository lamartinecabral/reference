- [**Algoritmos**](#algoritmos)
  - [Binary Search](#binary-search)
  - [Mo's Algorithm](#mos-algorithm)
  - [KMP](#kmp)
- [**Estruturas de Dados**](#estruturas-de-dados)
  - [Fenwick Tree (BIT)](#fenwick-tree-bit)
  - [Union Find](#union-find)
  - [Sparse Table](#sparse-table)
  - [SQRT Decomposition](#sqrt-decomposition)
  - [Ordered Set](#ordered-set)
  - [Segment Tree](#segment-tree)
    - [Lazy Propagation](#lazy-propagation)
    - [Dynamic Seg Tree](#dynamic-seg-tree)
    - [Dynamic Seg Tree 2D](#dynamic-seg-tree-2d)
  - [Trie](#trie)
	- [XOR trie](#xor-trie)
  - [Suffix Array & Longest Common Prefix Array](#suffix-array--longest-common-prefix-array)
  - [Wavelet Tree](#wavelet-tree)
  - [Treap](#treap)
  - [Minstack / Minqueue](#minstack--minqueue)
  - [BIT Variations](#bit-variations)
    - [BIT Range Update Range Query](#bit-range-update-range-query)
    - [BIT 2D](#bit-2d)
    - [BIT 2D Range Update Range Query](#bit-2d-range-update-range-query)
    - [Ordered Multiset with BIT](#ordered-multiset-with-bit)
  - [Gambiarras](#gambiarras)
    - [Ordered Multiset](#ordered-multiset)
    - [Bitset](#bitset)
    - [Inversed Vector](#inversed-vector)
	- [128bit Integer](#128bit-integer)
- [**Grafos**](#grafos)
  - [Busca em largura (BFS) e profundidade (DFS)](#busca-em-largura-bfs-e-profundidade-dfs)
  - [Dijkstra](#dijkstra)
  - [Floyd Warshall](#floyd-warshall)
  - [Spanning Tree (MST)](#spanning-tree-mst)
  - [Lowest Common Ancestor (LCA)](#lowest-common-ancestor-lca)
	- [LCA with Sparse Table](#lca-with-sparse-table)
    - [Binary Lifting & Query on Tree](#binary-lifting--query-on-tree)
  - [Topological Sort](#topological-sort)
  - [SCC Kosaraju](#scc-kosaraju)
  - [Travelling Salesman Problem (TSP)](#travelling-salesman-problem-tsp)
  - [Bipartite Matching (Kuhn's)](#bipartite-matching-kuhns)
  - [Maximum Flow](#maximum-flow)
  - [Min Cost Flow](#min-cost-flow)
- [**Programação Dinâmica**](#programação-dinâmica)
  - [Coin Change Problem](#coin-change-problem)
  - [Knapsack 0-1 Problem](#knapsack-0-1-problem)
    - [Backtracking Magic](#backtracking-magic)
  - [Longest Increasing Subsequence](#longest-increasing-subsequence)
  - [Digit DP](#digit-dp)
- [**Matemática**](#matemática)
  - [MDC e MMC](#mdc-e-mmc)
  - [Euclides Extendido](#euclides-extendido)
  - [Crivo de Eratostenes](#crivo-de-eratostenes)
  - [Exponenciação Rápida](#exponenciação-rápida)
  - [Fatoração](#fatoração)
  - [Totiente de Euler](#totiente-de-euler)
  - [Inverso Multiplicativo](#inverso-multiplicativo)
  - [Matrizes](#matrizes)
  - [Miller-Rabin's Prime Check & Pollard Rho's Algorithm](#miller-rabins-prime-check--pollard-rhos-algorithm)
  - [Fast Fourier Transform (FFT)](#fast-fourier-transform-fft)
  - [Congruência Linear](#congruência-linear)
  - [Teorema do Resto Chinês](#teorema-do-resto-chinês)
- [**Geometria**](#geometria)
  - [Pontos e Linhas](#pontos-e-linhas)
  - [Biblioteca Completa](#biblioteca-completa)
- [**Misc**](#misc)
  - [Codeforces Filter Script](#codeforces-filter-script)
  - [Visual Code Settings](#visual-code-settings)
  - [Bash Script](#bash-script)
  - [Fast IO](#fast-io)
  - [Template](#template)

# Algoritmos

### Binary Search

```c
bool test(int m){}

int binarySearch(int L, int R){ // [L,R]
	while(L<=R){
		int m = (L+R)/2;
		if(test(m)) L = m+1;
		else R = m-1;
	}
	return L; // primeiro false
	return R; // ultimo true
}

int binarySearch2(int L, int R){ // [L,R)
	while(L<R){
		int m = (L+R)/2;
		if(test(m)) L = m+1;
		else R = m;
	}
	return L-1; // ultimo true
	return L; // primeiro false
}
```

### Mo's Algorithm

```c
int block;
vector<int> v;
vector<array<int,3> > query; // {L,R,index}
int answer[100010];

void insere(int i){ }
void apaga(int i){ }
int solve(){ }

void mos(){
	block = sqrt(v.size());
	sort(all(query), [](array<int,3> &x, array<int,3> &y){
		if(x[0]/block != y[0]/block)
			return x[0] < y[0];
		return (x[0]/block)&1 ? x[1] > y[1] : x[1] < y[1];
	});
	
	int cl = 0, cr = 0; insere(0);
	for(auto &q: query){
		while(q[1] > cr) insere(++cr);
		while(q[1] < cr) apaga(cr--);
		while(q[0] < cl) insere(--cl);
		while(q[0] > cl) apaga(cl++);
		answer[q[2]] = solve();
	}
}
```

### KMP

```c
string s1,s2;
vector<int> b;

void kmppp(){
	int i=0, j=-1;
	b = vector<int>(s2.size()+1);
	b[0] = -1;
	while(i < s2.size()){
		while(j >= 0 && s2[i] != s2[j]) j=b[j];
		i++; j++;
		b[i] = j;
}}
void kmp(){
	int i=0, j=0;
	while(i < s1.size()){
		while(j >= 0 && s1[i] != s2[j]) j=b[j];
		i++; j++;
		if(j == s2.size()){
			cout<<"s2 found in s1 at "<< i-j <<endl;
			j=b[j];
}}}
```
# Estruturas de Dados

### Fenwick Tree (BIT)

```c
#define SZ 100010
int bit[SZ];

int qry(int i) {
	int sum = 0;
	while(i){
		sum += bit[i];
		i -= i&-i;
	}
	return sum;
}
void upd(int i, int k){
	while(i < SZ){
		bit[i] += k;
		i += i&-i;
	}
}
```

### Union Find

```c
struct Dsu{
	vector<int> g;

	Dsu(int n){
		g = vector<int>(n);
		for(int i=0; i<n; i++) g[i] = i;
	}

	int find(int i){
		return group[i] == i ? i : g[i] = find(g[i]);
	}

	void union(int i, int j){
		g[find(i)] = find(j);
	}
};
```

### Sparse Table

```c
struct SparseTable{
	vector< array<int,17> > st; // st[1000000][17]
	SparseTable(int* bg, int* en){
		int n = en-bg; st.assign(n);
		for(int i=0; i<n; i++) st[i][0] = bg[i];
		for(int j = 1; (1<<j) <= n; j++)
			for(int i=0; i+(1<<j) <= n; i++)
				st[i][j] = min(st[i][j-1], st[i+(1<<(j-1))][j-1]);
	}
	int query(int l, int r){
		int k = log2(r-l+1);
		return min(st[l][k], st[r+1-(1<<k)][k]);
	}
};
```

### SQRT Decomposition

```c
int n,raiz;
int v[100010];
int bl[325];

void build(){
	raiz = sqrt(n);
	for(int i=0; i<n; i++) bl[i/raiz] += v[i];
}
int query(int l, int r){
	ll sum = 0;
	for(; l<r and l%raiz!=0; l++){
		sum += v[l];
	} for(; l+raiz <= r; l += raiz){
		sum += bl[l/raiz];
	} for(; l<=r; l++){
		sum += v[l];
	}
	return sum;
}
void pointupdate(int i, int k){
	int b = i/raiz;
	bl[b] += k - v[i];
	v[i] = k;
}
```

### Ordered Set

```c
#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp>
using namespace __gnu_pbds;

#define ordered_set tree<int,null_type,less<int>,rb_tree_tag,tree_order_statistics_node_update>

// find_by_order(k) = pointer to kth element (0-based)
// order_of_key(x) = how many elements less than x
```

### Segment Tree

```c
typedef int var;

int k;
var st[400040];
var v[100010];

var neutro = 0;
var combine(var a, var b){
	return a+b;
}

void build(int n){
	k = n;
	while(k != (k&-k) )
		k += k&-k;
	
	for(int i=0; i<k+k; i++)
		st[i] = neutro;
	
	for(int i=0; i<n; i++)
		st[k+i] = v[i];
	
	for(int i=k-1; i; i--)
		st[i] = combine( st[2*i], st[2*i+1] );
}

int L,R;
var qry(int i, int l, int r){
	if(l >= L && r <= R) return st[i];
	if(r < L || l > R) return neutro;
	int mid = (l+r)/2;
	return combine( qry(i*2,l,mid) , qry(i*2+1,mid+1,r) );
}

var query(int l, int r){
	L = l, R = r;
	return qry(1,0,k-1);
}

void update(int i, var x){
	v[i] = x;
	i += k;
	st[i] = x;
	for(i /= 2; i; i /= 2)
		st[i] = combine( st[i*2], st[i*2+1] );
}

// 0-indexed
// left and right included on interval
// How to use: build(size); query(left,right); update(index,value);
```

##### Lazy Propagation

```c
const int neutral = 0; //comp(x, neutral) = x
int comp(int a, int b) {
	return a + b;
}

struct SegmentTree {
	vector<int> st, lazy;
	int size;
#define esq(p) (p << 1)
#define dir(p) ((p << 1) + 1)
	void build(int p, int l, int r, int* A) {
		if (l == r) { st[p] = A[l]; return; }
		int m = (l + r) / 2;
		build(esq(p), l, m, A);
		build(dir(p), m+1, r, A);
		st[p] = comp(st[esq(p)], st[dir(p)]);
	}
	void push(int p, int l, int r) {
		st[p] += (r - l + 1)*lazy[p];	//Caso RSQ
		//st[p] += lazy[p]; 		    //Caso RMQ
		if (l != r) {
			lazy[dir(p)] += lazy[p];
			lazy[esq(p)] += lazy[p];
		}
		lazy[p] = 0;
	}
	void update(int p, int l, int r, int a, int b, int k) {
		push(p, l, r);
		if (a > r || b < l) return;
		else if (l >= a && r <= b) {
			lazy[p] = k; push(p, l, r); return;
		}
		int m = (l + r) / 2;
		update(esq(p), l, m, a, b, k);
		update(dir(p), m+1, r, a, b, k);
		st[p] = comp(st[esq(p)], st[dir(p)]);
	}
	int query(int p, int l, int r, int a, int b) {
		push(p, l, r);
		if (a > r || b < l) return neutral;
		if (l >= a && r <= b) return st[p];
		int m = (l + r) / 2;
		int p1 = query(esq(p), l, m, a, b);
		int p2 = query(dir(p), m+1, r, a, b);
		return comp(p1, p2);
	}

	SegmentTree(int* bg, int* en) {
		size = (int)(en - bg);
		st.assign(4 * size, neutral);
		lazy.assign(4 * size, 0);
		build(1, 0, size - 1, bg);
	}
	int query(int a, int b) { return query(1, 0, size - 1, a, b); }
	void update(int a, int b, int k) { update(1, 0, size - 1, a, b, k); }
};
```

##### Dynamic Seg Tree

```c
const int SZ = 4e6;
int st[SZ], child[SZ][2], size = 2;

int minl = -1e9-10; int maxr = 1e9+10; // [l,r)
int neutro = 0;
int combine(int x, int y){ return x+y; }

int I,X;
void update(int i, int l, int r){
	if(I == l && I == r-1){ st[i] += X; return; }
	int m = (l+r)/2;
	int k = I >= m;
	if(child[i][k] == 0) child[i][k] = size++;
	update(child[i][k], k?m:l, k?r:m);
	st[i] = combine( st[child[i][0]] , st[child[i][1]] );
}
void update(int i, int x){ I = i; X = x; update(1,minl,maxr); }

int L,R; // [L,R)
int query(int i, int l, int r){
	if(i == 0) return neutro;
	if(L <= l && r <= R) return st[i];
	if(R <= l || r <= L) return neutro;
	int m = (l+r)/2;
	return combine( query(child[i][0],l,m), query(child[i][1],m,r) );
}
int query(int l, int r){ L = l, R = r; return query(1,minl,maxr); }
```

##### Dynamic Seg Tree 2D

```c
const int SZ = 4e6;
int st[SZ], child[SZ][4], size = 2;

int neutro = 0; // edit here
int combine(int a, int b){ return a+b; }
int minn = -1e2; int maxx = 1e2; // edit here

int W,X,Y;
void update(int i, int x0, int y0, int x1, int y1){
	if( X == x0 && X == x1-1 && Y == y0 && Y == y1-1){
		st[i] += W; return; }
	int mx = (x0+x1)/2;
	int my = (y0+y1)/2;
	int k = 0;
	if(X >= mx) k += 1;
	if(Y >= my) k += 2;
	if(child[i][k] == 0) child[i][k] = size++;
	update(child[i][k], (k&1)?mx:x0, (k&2)?my:y0, (k&1)?x1:mx, (k&2)?y1:my);
	
	st[i] = neutro;
	for(int k=0; k<4; k++)
		st[i] = combine( st[i], st[child[i][k]] );
}
void update(int w, int x, int y){
	X = x; Y = y; W = w; update(1,minn,minn,maxx,maxx);
}

int X0,Y0,X1,Y1; // not including line x1 and column y1
int query(int i, int x0, int y0, int x1, int y1){
	if(i == 0) return neutro;
	if( X0 <= x0 && Y0 <= y0 && x1 <= X1 && y1 <= Y1) return st[i];
	if( X1 <= x0 || Y1 <= y0 || x1 <= X0 || y1 <= Y0) return neutro;
	int mx = (x0+x1)/2;
	int my = (y0+y1)/2;
	
	int res = neutro;
	for(int k=0; k<4; k++) res = combine(res,
		query(child[i][k], (k&1)?mx:x0, (k&2)?my:y0, (k&1)?x1:mx, (k&2)?y1:my)
	); return res;
}
int query(int x0, int y0, int x1, int y1){
	X0 = x0, Y0 = y0; X1 = x1, Y1 = y1; return query(1,minn,minn,maxx,maxx);
}
```

### Trie

```c
struct trienode{
	int child[26];
	int isleaf;
	trienode(){
		memset(child,0,sizeof child);
		isleaf = 0;
	}
};
vector<trienode> trie;

void insere(string &s){
	int i = 0;
	for(auto x: s){
		int c = x-'a';
		if(trie[i].child[c] == 0) {
			trie[i].child[c] = trie.size();
			trie.push_back(trienode());
		}
		i = trie[i].child[c];
	}
	trie[i].isleaf++;
}

bool busca(string &s){
	int i = 0;
	for(auto x: s){
		int c = x-'a';
		if(trie[i].child[c] != 0)
			i = trie[i].child[c];
		else return false;
	}
	return trie[i].isleaf;
}
```

##### XOR trie

```c
struct trienode {
	int child[2], size;
	trienode(){ memset(child,0,sizeof child); size = 0; }
};
vector<trienode> trie;
void init(){ trie.clear(); trie.push_back(trienode()); }
void insere(int x){
	int i = 0;
	for(int k = 30; k >= 0; k--){
		int c = x&(1<<k) ? 1 : 0;
		if(trie[i].child[c] == 0){
			trie[i].child[c] = trie.size();
			trie.push_back(trienode()); }
		i = trie[i].child[c];
		trie[i].size++;
}}
int qry(int x){
	int i = 0, ret = 0;
	for(int k = 30; k >= 0; k--){
		int c = x&(1<<k) ? 1 : 0;
		if(trie[trie[i].child[1-c]].size) // maximizar (qry(x)^x)
		//if(trie[trie[i].child[c]].size == 0) // minimizar (qry(x)^x)
			c = 1-c;
		i = trie[i].child[c];
		ret |= c<<k;
	} return ret;
}
bool busca(int x){
	int i = 0;
	for(int k = 30; k >= 0; k--){
		int c = x&(1<<k) ? 1 : 0;
		i = trie[i].child[c];
		if(trie[i].size == 0) return false;
	} return true;
}
void remove(int x){ if(!busca(x)) return;
	int i = 0;
	for(int k = 30; k >= 0; k--){
		int c = x&(1<<k) ? 1 : 0;
		i = trie[i].child[c];
		trie[i].size--;
}}
```

### Suffix Array & Longest Common Prefix Array

```c
const int MAXN = 100010;
char str[MAXN]; // the input string, up to 100K characters
int n; // the length of input string
int RA[MAXN], tempRA[MAXN]; // rank array and temporary rank array
int SA[MAXN], tempSA[MAXN]; // suffix array and temporary suffix array
int c[MAXN]; // for counting/radix sort

void countingSort(int k) { // O(n)
	int i, sum, maxi = max(300, n);
	memset(c, 0, sizeof c);
	for (i = 0; i < n; i++)  c[i + k < n ? RA[i + k] : 0]++;
	for (i = sum = 0; i < maxi; i++) {
		int t = c[i];
		c[i] = sum;
		sum += t;
	}
	for (i = 0; i < n; i++)  tempSA[c[SA[i]+k < n ? RA[SA[i]+k] : 0]++] = SA[i];
	for (i = 0; i < n; i++)  SA[i] = tempSA[i];
}

void constructSA() { //O(nlogn)
	int i, k, r;
	for (i = 0; i < n; i++) RA[i] = str[i];
	for (i = 0; i < n; i++) SA[i] = i;
	for (k = 1; k < n; k <<= 1) {
		countingSort(k);
		countingSort(0);
		tempRA[SA[0]] = r = 0;
		for (i = 1; i < n; i++)  tempRA[SA[i]] =
			(RA[SA[i]] == RA[SA[i-1]] && RA[SA[i]+k] == RA[SA[i-1]+k]) ? r : ++r;
		for (i = 0; i < n; i++)  RA[i] = tempRA[i];
		if (RA[SA[n-1]] == n-1) break;
	}
}

int Phi[MAXN];
int LCP[MAXN], PLCP[MAXN];

//Longest Common Prefix
//LCP[i] keeps the size of the longest common prefix between SA[i] and SA[i-1]
void computeLCP() { //O(n)
	int i, L;
	Phi[SA[0]] = -1;
	for (i = 1; i < n; i++)  Phi[SA[i]] = SA[i-1];
	for (i = L = 0; i < n; i++) {
		if (Phi[i] == -1) {
			PLCP[i] = 0; continue;
		}
		while (str[i + L] == str[Phi[i] + L]) L++;
		PLCP[i] = L;
		L = max(L-1, 0);
	}
	for (i = 0; i < n; i++) LCP[i] = PLCP[SA[i]];
}
```

### Wavelet Tree

```c
const int MAX = 1e6;
struct wavelet_tree{
	int lo, hi;
	wavelet_tree *l, *r;
	vi b;

	//nos are in range [x,y]
	//array indices are [from, to)
	//init: wavelet_tree T(arr+1, arr+n+1, 1, MAX);
	wavelet_tree(int *from, int *to, int x, int y){
		lo = x, hi = y;
		if(lo == hi or from >= to) return;
		int mid = (lo+hi)/2;
		auto f = [mid](int x){
			return x <= mid;
		};
		b.reserve(to-from+1);
		b.pb(0);
		for(auto it = from; it != to; it++)
			b.pb(b.back() + f(*it));
		//see how lambda function is used here	
		auto pivot = stable_partition(from, to, f);
		l = new wavelet_tree(from, pivot, lo, mid);
		r = new wavelet_tree(pivot, to, mid+1, hi);
	}

	//kth smallest element in [l, r].
	//k belongs to [1,r-l+1]
	int kth(int l, int r, int k){
		if(l > r) return 0;
		if(lo == hi) return lo;
		int inLeft = b[r] - b[l-1];
		int lb = b[l-1]; //amt of nos in first (l-1) nos that go in left 
		int rb = b[r]; //amt of nos in first (r) nos that go in left
		if(k <= inLeft) return this->l->kth(lb+1, rb , k);
		return this->r->kth(l-lb, r-rb, k-inLeft);
	}

	//count of nos in [l, r] Less than or equal to k
	int LTE(int l, int r, int k) {
		if(l > r or k < lo) return 0;
		if(hi <= k) return r - l + 1;
		int lb = b[l-1], rb = b[r];
		return this->l->LTE(lb+1, rb, k) + this->r->LTE(l-lb, r-rb, k);
	}

	//count of nos in [l, r] equal to k
	int count(int l, int r, int k) {
		if(l > r or k < lo or k > hi) return 0;
		if(lo == hi) return r - l + 1;
		int lb = b[l-1], rb = b[r], mid = (lo+hi)/2;
		if(k <= mid) return this->l->count(lb+1, rb, k);
		return this->r->count(l-lb, r-rb, k);
	}
	~wavelet_tree(){
		delete l;
		delete r;
	}
};
```

### Treap

```c
struct Node {
	int key, prior, size;
	Node *l, *r;

	Node(int _key) : key(_key), prior(rand()), size(1), l(NULL), r(NULL) {}
	~Node() { delete l; delete r; }

	void recalc() {
		size = 1;
		if (l) size += l->size;
		if (r) size += r->size;
	}
};

struct Treap {

	Node * merge(Node * l, Node * r) {
		if (!l || !r) return l ? l : r;
		// Se a prioridade esquerda é menor.
		if (l->prior < r->prior) {
			l->r = merge(l->r, r);
			l->recalc();
			return l;
			// Se a prioridade direita é maior ou igual.
		} else {
			r->l = merge(l, r->l);
			r->recalc();
			return r;
		}
	}

	// keys maiores ou iguais a "key" ficarão no r, e os demais no l.
	void split(Node * v, int key, Node *& l, Node *& r) {
		l = r = NULL;
		if (!v) return;
		// Se o key for maior, ir para direita
		if (v->key < key) {
			split(v->r, key, v->r, r);
			l = v;
			// Se o key for menor ou igual ir para esquerda.
		} else {
			split(v->l, key, l, v->l);
			r = v;
		}
		v->recalc();
	}

	bool find(Node *v, int key){
		if(!v) return false;
		if( v->key == key ) return true;
		if( v->key < key ) return find(v->r, key);
		if( v->key > key ) return find(v->l, key);
	}

	int smallerCount(Node *v, int key){
		if(!v) return 0;
		// Se for menor ou igual adicionar + 1.
		if( v->key == key ) return ( v->l ? v->l->size : 0 );
		if( v->key < key )
			return 1 + ( v->l ? v->l->size : 0 )
				+ smallerCount(v->r, key);
		if( v->key > key ) return smallerCount(v->l, key);
	}

	Node * kth(Node *v, int k){
		if(!v) return NULL;
		int left = (v->l? v->l->size: 0);
		if( k-left == 1 ) return v;
		if( k-left > 1 ) return kth(v->r, k-left-1);
		if( k-left < 1 ) return kth(v->l, k);
	}

	Node * root;
	Treap() : root(NULL) {}
	~Treap() { delete root; }

	// Se existe um elemento com o key
	bool find(int key){
		return find(root, key);
	}
	// Quantos elementos menores que key
	int smallerCount(int key){
		return smallerCount(root, key);
	}
	// Quantos elementos iguais a key
	int count(int key){
		return smallerCount(root,key+1)-smallerCount(root,key);
	}
	// Retorna o k-th menor elemento
	// k deve pertencer a [1,size]
	int kth(int k){
		auto v = kth(root, k);
		return v ? v->key : INF;
	}
	// Insere o key mesmo se já exista outro com key igual
	void insert(int key) {
		Node * l, * r;
		split(root, key, l, r);
		root = merge(merge(l, new Node(key)), r);
	}
	// Apaga todos os elementos que possuem o key.
	void erase(int key) {
		Node * l, * m, * r;
		split(root, key, l, m);
		split(m, key + 1, m, r);
		delete m;
		root = merge(l, r);
	}
	int size() const { return root ? root->size : 0; }
};
```

### Minstack / Minqueue

```c
struct minstack{
	vector< pair<int,int> > s;
	bool empty(){ return s.empty(); }
	int size(){ return s.size(); }
	int top(){ return s.back().fi; }
	int mini(){ return s.back().se; }
	void pop(){ return s.pop_back(); }
	void push(int x){
		s.push_back({ x, s.empty() ? x : min(x, s.back().se) }); }
};

struct minqueue{
	minstack s1,s2;
	void transfer(){ while(!s1.empty()){ s2.push(s1.top()); s1.pop(); } }
	bool empty(){ return s1.empty() && s2.empty(); }
	int size(){ return s1.size()+s2.size(); }
	void pop(){ if(s2.empty()) transfer(); s2.pop(); }
	int front(){ if(s2.empty()) transfer(); return s2.top(); }
	void push(int x){ s1.push(x); }
	int mini(){
		if(s2.empty()) transfer();
		else if(!s1.empty()) return min(s1.mini(),s2.mini());
		return s2.mini();
	}
};
```

### BIT Variations

##### BIT Range Update Range Query

```c
#define SZ 100010
int bit1[SZ], bit2[SZ];

// retorna o somatorio a[1]+a[2]+...+a[i]
int query(int idx){
	int sum1=0, sum2=0;
	for(int i = idx; i; i -= i&-i) sum1 += bit1[i];
	for(int i = idx; i; i -= i&-i) sum2 += bit2[i];
	return sum1*idx - sum2;
}

// incrementa x em todos a[i],a[i+1],...,a[N]
void update(int idx, int x){
	for(int i = idx; i<SZ; i += i&-i) bit1[i] += x;
	for(int i = idx; i<SZ; i += i&-i) bit2[i] += x*(idx-1);
}
```

##### BIT 2D

```c
#define SZ 1010
int bit[SZ][SZ];

int query(int idx, int jdx){
	int sum = 0;
	for(int i=idx; i; i -= i&-i)
		for(int j=jdx; j; j -= j&-j)
			sum += bit[i][j];
	return sum;
}
void update(int idx, int jdx, int k){
	for(int i=idx; i<SZ; i += i&-i)
		for(int j=jdx; j<SZ; j += j&-j)
			bit[i][j] += k;
}
// para point update range query
var rangequery(int x1, int y1, int x2, int y2){
	return query(x2,y2) - query(x2,y1-1) - query(x1-1,y2) + query(x1-1,y1-1);
}
// para range update point query
void rangeupdate(int x1, int y1, int x2, int y2, int k){
	update(x1,y1,k); update(x2+1,y1,-k); update(x1,y2+1,-k); update(x2+1,y2+1,k);
}
```

##### BIT 2D Range Update Range Query

```c
#define SZ 1010
int bit[4][SZ][SZ];

void upd(int id, int x, int y, int k){
	for(int i = x; i<SZ; i += i&-i)
		for(int j = y; j<SZ; j += j&-j)
			bit[id][i][j] += k;
}

int qry(int id, int x, int y){
	int sum = 0;
	for(int i = x; i>0; i -= i&-i)
		for(int j = y; j>0; j -= j&-j)
			sum += bit[id][i][j];
	return sum;
}

void update(int x, int y, int k){
	upd(0, x, y, k);
	upd(1, x, y, k*(x-1) );
	upd(2, x, y, k*(y-1) );
	upd(3, x, y, k*(x-1)*(y-1) );
}

int query(int x, int y){
	int sum = 0;
	sum += qry(0,x,y)*x*y;
	sum -= qry(1,x,y)*y;
	sum -= qry(2,x,y)*x;
	sum += qry(3,x,y);
	return sum;
}

int rangequery(int x1, int y1, int x2, int y2){
	return query(x2,y2) - query(x2,y1-1) - query(x1-1,y2) + query(x1-1,y1-1);
}

void rangeupdate(int x1, int y1, int x2, int y2, int k){
	update(x1,y1,k); update(x2+1,y1,-k); update(x1,y2+1,-k); update(x2+1,y2+1,k);
}
```

##### Ordered Multiset with BIT

```c
struct ordered_multiset{
	
	vector<int> bit; int contador, N, LOGN, fix;
	ordered_multiset(int mini, int maxi){
		N = maxi - mini + 2;
		fix = 1 - mini;
		bit.assign(N,0); contador = 0; LOGN = log2(N)+1; }
	
	int size(){
		return contador;
	}
	void insert(int x){ x += fix; // must be inside [1,N-1]
		for(int i=x; i<N; i += i&-i) bit[i]++; contador++;
	}
	void erase(int x){ x += fix; // set must contain x
		for(int i=x; i<N; i += i&-i) bit[i]--; contador--;
	}
	int lte(int x){ x += fix; // how many less than or equal
		int sum = 0;
		for(int i=x; i; i -= i&-i) sum += bit[i];
		return sum;
	}
	int kth(int k){ // must be inside [1,N-1]
		int sum = 0; int pos = 0;
		for(int i=LOGN; i>=0; i--)
			if(pos + (1 << i) < N && sum + bit[pos + (1 << i)] < k){
				sum += bit[pos + (1 << i)];
				pos += (1 << i); }
		return pos + 1 - fix;
	}
};
```

### Gambiarras

##### Ordered Multiset

```c
#define pii pair<int,int>
#define ordered_set tree<pii,null_type,less<pii>,rb_tree_tag,tree_order_statistics_node_update>

int id = 0; map<int, vector<int> > ids;
void insere(ordered_set &s, int x){
	s.insert({x,++id}); ids[x].pb(id);
}
void apaga(ordered_set &s, int x){ if(ids[x].size()==0) return;
	s.erase({x,ids[x].back()}); ids[x].pop_back();
}
int kth(ordered_set &s, int x){
	return s.find_by_order(x)->fi;
}
int smallerCount(ordered_set &s, int x){
	return s.order_of_key({x,0});
}
int count(ordered_set &s, int x){
	return smallerCount(s,x+1)-smallerCount(s,x);
}
ordered_set::iterator find(ordered_set &s, int x){ if(ids[x].size()==0) return s.end();
	return s.find({x,ids[x].back()});
}
```

##### Bitset

```c
const int N = 1e8; int bs[N/32];
bool bget(int i){
	return bs[i>>5]&(1<<(i&31));
}
void bset(int i){
	bs[i>>5] |= (1<<(i&31));
}
void breset(int i){
	bs[i>>5] &= ~(1<<(i&31));
}
```

##### Inversed Vector

```c
struct ivi{ // inversed_vector<int>
	vector<int> a;
	ivi operator=(vi v) { a = v; reverse(all(a)); return (*this); }
	int& operator[](int i){ return a[a.size()-1-i]; }
	void push_front(int x){ a.push_back(x); }
	int size(){ return (int)a.size(); }
	void swap(ivi& b){ a.swap(b.a); }
};
```

##### 128bit Integer

```c
#define lll __int128_t
ostream &operator<<(ostream &os, lll n){
	int s = n<0 ? -1 : 1; n*=s;
	long long x = 1e13; deque<long long> v;
	while(n){ v.push_front(1LL*(n%x)); n/=x; }
	for(long long i:v){ os<<s*i; s*=s; } return os; }
```

# Grafos

### Busca em largura (BFS) e profundidade (DFS)

```c
#define SZ 100010
vector<int> g[SZ];
bool vis[SZ];

void bfs(int start){
	queue<int> q;
	q.push(start);
	vis[start] = true;
	while(!q.empty()){
		int v = q.front();
		q.pop();
		for(auto u: g[v]){
			if(vis[u] == false){
				vis[u] = true;
				q.push(u);
}}}}

void dfs(int v){
	vis[v] = true;
	for(auto u: g[v]){
		if(vis[u] == false){
			dfs(u);
}}}
```

### Dijkstra

```c
#define SZ 100010
vector<pii> g[SZ];
int d[SZ];

void dijkstra(int start){
	memset(d,INF,sizeof d); d[start] = 0;

	typedef array<int,2> vet;
	priority_queue< vet, vector<vet>, greater<vet> > pq;
	pq.push( {0,start} );
	while(!pq.empty()){
		int dv = pq.top()[0];
		int v = pq.top()[1];
		pq.pop();
		if(dv > d[v]) continue;
		for(auto x: g[v]){
			int u = x.fi;
			int w = x.se;
			if(d[u] > dv + w){
				d[u] = dv + w;
				pq.push( {d[u], u} );
}}}}
```

### Floyd Warshall

```c
// distance matrix
#define SZ 300
int g[SZ][SZ];

void floyd(int n){
	for(int k=0; k<n; k++)
		for(int i=0; i<n; i++)
			for(int j=0; j<n; j++)
				g[i][j] = min(g[i][j], g[i][k]+g[k][j] );
}
```

### Spanning Tree (MST)

```c
#define SZ 100010
vector<pii> g[SZ],h[SZ];
	
void prim(){
	priority_queue< array<int,3> > pq;
	for(auto p: g[1]) pq.push( {-p.se, p.fi, 1} );
	bool vis[SZ]; memset(vis,0,sizeof vis); vis[1] = 1;
	while(!pq.empty()){
		int v = pq.top()[1], u = pq.top()[2];
		int d = -pq.top()[0]; pq.pop();
		if(vis[v]) continue; else vis[v]=1;
		h[v].push_back({u,d}); h[u].push_back({v,d});
		for(auto& p: g[v]) pq.push({-p.se, p.fi, v});
	}
}
```

### Lowest Common Ancestor (LCA)

##### LCA with Sparse Table

```c
#define SZ 100010
vector<int> g[SZ];
int cont;
int id[SZ]; // ultimo contador do vertice
vector< pair<int,int> > arr; // array pro lca

void linear(int v, int ant, int altura){
	id[v] = cont++;
	arr.pb({altura,v});
	for(auto u: g[v]) if(u!=ant){
		linear(u, v, altura+1);
		id[v] = cont++;
		arr.pb({altura,v});
	}
}

pair<int,int> st[SZ*2][21]; // must be [n][logn+1]
void buildtable(){
	int n = arr.size(); int m = log2(n-1)+1;
	FOR(i,0,n) st[i][0] = arr[i];	
	for(int j=1,p=2; j<m; j++,p*=2) FOR(i,0,n-p+1)
		st[i][j] = min( st[i][j-1], st[i+p/2][j-1] );
}

void build(int raiz){
	cont = 0; arr.clear();
	linear(raiz,raiz,-1);
	buildtable();
}

int lca(int i, int j){
	i = id[i]; j = id[j];
	if(i==j) return st[i][0].se;
	if(i>j) swap(i,j);
	int k = log2(j-i);
	return min(st[i][k], st[j+1-(1<<k)][k]).se;
}
```

##### Binary Lifting & Query on Tree

```c
int n,m;
vector<pii> g[SZ];
#define LOGN 20
int parent[SZ][LOGN];
int height[SZ];
int query[SZ][LOGN]; //q[i][j] = maior aresta no caminho entre o vertice i e o ancestor 2^j

void init(int v, int p, int h, int q){
	parent[v][0] = p; height[v] = h; query[v][0] = q;
	for(auto &u: g[v]) if(u.fi!=p) init(u.fi,v,h+1,u.se);
}
void build(){ // é importante que os vertices sejam 1-based e o pai da raiz seja 0
	init(1,0,0,-INF); // raiz,pai,altura,weight(v->p)
	FOR(k,1,LOGN) FOR(v,1,n+1){
		int p = parent[v][k-1];
		parent[v][k] = parent[p][k-1];
		query[v][k] = max(query[v][k-1],query[p][k-1]); // edit here
}}
int lca(int v, int u){
	if(height[v] < height[u]) swap(v,u);
	int x = height[v]-height[u];
	FOR(k,0,LOGN) if(x&(1<<k)) v = parent[v][k]; // walk x
	if(v==u) return v;
	for(int k=LOGN-1; k>=0; k--) if(parent[v][k]!=parent[u][k]){
		v = parent[v][k]; u = parent[u][k];
	} return parent[v][0];
}
int qry(int v, int u){
	vi vertices = {v,u}; int ans = -INF; // edit here
	for(auto t: vertices){
		int x = height[t]-height[lca(v,u)];
		FOR(k,0,LOGN) if(x&(1<<k)){
			ans = max(ans,query[t][k]); // edit here
			t = parent[t][k];
	}} return ans;
}
```

### Topological Sort

Se a ordenação topológica for única
então ela forma um caminho hamiltoniano,
senão um caminho hamiltoniano não existe.

##### DFS
```c
vi g[SZ], ts;

bool vis[SZ];
void dfs(int u){
	vis[u] = true;
	for(auto v: g[u])
		if(!vis[v]) dfs(v);
	ts.push_back(u);
}
void Topological(int n){
	FOR(i,0,n) if(!vis[i]) dfs(i);
	reverse(all(ts));
}
```

##### BFS
```
vi g[SZ];

vi Topological(int n){
	vi res; int grau[n] = {0};
	FOR(v,0,n) for(auto u: g[v]) grau[v]++;
	priority_queue<int> pq;
	FOR(v,0,n) if(grau[v] == 0) pq.push(-v);
	while(pq.size()){
		int v = -pq.top(); pq.pop(); res.pb(v);
		for(auto u: g[v]) if(--grau[u] == 0) pq.push(-u);
	}
	return res;
}
```

### SCC Kosaraju

```c
// http://www.geeksforgeeks.org/strongly-connected-components/
vi g[SZ], h[SZ]; // grafos indo e voltando
bool vis[SZ];
vi res,temp;
stack<int> s;

void dfs(int v){
	vis[v] = 0;
	for(int i=0; i<g[v].size(); i++){
		if(vis[g[v][i]]) dfs(g[v][i]);
	} s.push(v);
}

void dfs2(int v){
	vis[v] = 0;
	for(int i=0; i<h[v].size(); i++){
		if(vis[h[v][i]]) dfs2(h[v][i]);
	} temp.pb(v);
}

void kosaraju(){
	memset(vis,1,sizeof vis);
	for(int i=1; i<=n; i++){
		if(vis[i]) dfs(i);
	}
	memset(vis,1,sizeof vis); res.clear();
	while(!s.empty()){
		temp.clear();
		if(vis[s.top()]) dfs2(s.top());
		if(temp.size() > res.size()) res = temp; // maior scc
		s.pop();
	}
}
```

### Travelling Salesman Problem (TSP)

TSP reduzido pra calcular o custo da melhor permutação.
###### Iterativo:

```c
int N;
double h[N][N];
double tsp[1<<N][N];
// tsp[S][i] = o custo de visitar todos os
// nós de S onde i foi o ultimo nó visitado

void build(int ori){
	memset(tsp,INF,sizeof tsp);
	for(int i=0; i<N; i++) if(i!=ori) tsp[1<<i][i] = h[ori][i];
	
	for(int bit=0; bit<(1<<N); bit++)
	for(int i=0; i<N; i++) if( bit&(1<<i) )
	for(int j=0; j<N; j++) if( bit&(1<<j) )
	if( i!=j )
	tsp[bit][i] = min( tsp[bit][i], tsp[bit^(1<<i)][j] + h[j][i] );
}
```

###### Recursivo:

```c
int N, ori, h[MAXN][MAXN], pd[1<<MAXN][MAXN];
// pd[S][i] = o custo de visitar todos os
// nós de S onde i foi o ultimo nó visitado

int tsp(int bit, int i){
	if(pd[bit][i]!=-1) return pd[bit][i];
	if(bit == (1<<i)){
		pd[bit][i] = h[ori][i];
		return pd[bit][i]; }
	
	pd[bit][i] = INF;
	for(int j=0; j<N; j++) if(bit&(1<<j) && i!=j)
		pd[bit][i] = min(pd[bit][i], tsp(bit^(1<<i),j)+h[j][i] );
	return pd[bit][i];
}
```

### Bipartite Matching (Kuhn's)

```c
int vis[SZ], b[SZ], tempo;
vi g[SZ]; //1-based

bool kuhn(int u){
	if(vis[u]==tempo) return false;
	vis[u] = tempo;
	for(int v: g[u]){
		if( !b[v] || kuhn(b[v]) ){
			b[v] = u;
			return true;
		}
	}
	return false;
}

int matching(int A, int B){
	for(int i=1; i<=A; i++){
		tempo++; kuhn(i);
	}
	int cont = 0;
	for(int i=1; i<=B; i++) if(b[i]) cont++;
	return cont;
}
```

### Maximum Flow

```c
// O(V*V*E)
const int MAXN = 1e5;
struct edge { int a, b, cap, flow; };

int n, s, t, d[MAXN], ptr[MAXN], q[MAXN];
vector<edge> e;
vector<int> g[MAXN];

void add_edge (int u, int v, int cap) {
	g[u].pb(e.size()); e.push_back({ u, v, cap, 0 });
	g[v].pb(e.size()); e.push_back({ v, u, 0, 0 });
}
 
bool bfs() {
	int qh=0, qt=0; q[qt++] = s;
	memset (d, -1, n * sizeof d[0]); d[s] = 0;
	while (qh < qt && d[t] == -1) {
		int v = q[qh++];
		for (int i=0; i<g[v].size(); ++i) {
			int id = g[v][i], to = e[id].b;
			if (d[to] == -1 && e[id].flow < e[id].cap) {
				q[qt++] = to;
				d[to] = d[v] + 1;
	}}} return d[t] != -1;
}
 
int dfs (int v, int flow) {
	if(!flow) return 0;
	if(v == t) return flow;
	for(; ptr[v]<(int)g[v].size(); ++ptr[v]) {
		int id = g[v][ptr[v]], to = e[id].b;
		if(d[to] != d[v] + 1)  continue;
		int pushed = dfs(to, min (flow, e[id].cap - e[id].flow));
		if(pushed) {
			e[id].flow += pushed;
			e[id^1].flow -= pushed;
			return pushed;
	}} return 0;
}
 
int dinic(){
	int flow = 0;
	while(bfs()){
		memset(ptr, 0, n * sizeof ptr[0]);
		while(int pushed = dfs (s, INF))
			flow += pushed;
	} return flow;
}

vector< array<int,3> > lista_de_arestas;
 
int calcula_fluxo(int quantos_vertices){
	n = quantos_vertices; s = 0; t = n-1;
	e.clear(); FOR(i,0,n) g[i].clear();
	for(auto a: lista_de_arestas)
		add_edge(a[0],a[1],a[2]);
	return dinic();
}
```

### Min Cost Flow

```c
struct Edge
{
    int from, to, capacity, cost;
};

vector<vector<int>> adj, cost, capacity;

const int INF = 1e9;

void shortest_paths(int n, int v0, vector<int>& d, vector<int>& p) {
    d.assign(n, INF);
    d[v0] = 0;
    vector<bool> inq(n, false);
    queue<int> q;
    q.push(v0);
    p.assign(n, -1);

    while (!q.empty()) {
        int u = q.front();
        q.pop();
        inq[u] = false;
        for (int v : adj[u]) {
            if (capacity[u][v] > 0 && d[v] > d[u] + cost[u][v]) {
                d[v] = d[u] + cost[u][v];
                p[v] = u;
                if (!inq[v]) {
                    inq[v] = true;
                    q.push(v);
                }
            }
        }
    }
}

int min_cost_flow(int N, vector<Edge> edges, int K, int s, int t) {
    adj.assign(N, vector<int>());
    cost.assign(N, vector<int>(N, 0));
    capacity.assign(N, vector<int>(N, 0));
    for (Edge e : edges) {
        adj[e.from].push_back(e.to);
        adj[e.to].push_back(e.from);
        cost[e.from][e.to] = e.cost;
        cost[e.to][e.from] = -e.cost;
        capacity[e.from][e.to] = e.capacity;
    }

    int flow = 0;
    int cost = 0;
    vector<int> d, p;
    while (flow < K) {
        shortest_paths(N, s, d, p);
        if (d[t] == INF)
            break;

        // find max flow on that path
        int f = K - flow;
        int cur = t;
        while (cur != s) {
            f = min(f, capacity[p[cur]][cur]);
            cur = p[cur];
        }

        // apply flow
        flow += f;
        cost += f * d[t];
        cur = t;
        while (cur != s) {
            capacity[p[cur]][cur] -= f;
            capacity[cur][p[cur]] += f;
            cur = p[cur];
        }
    }

    if (flow < K)
        return -1;
    else
        return cost;
}
```

# Programação Dinâmica

### Coin Change Problem

```c
vector<int> moeda;
int troco[maxt];
void build(){
	troco[0] = 0;
	for(int i=1; i<N; ++i){
		troco[i] = 1e9;
		for(int x: moeda)
			if(x<=i) troco[i] = min(troco[i], troco[i-x]+1 );
}}
```

### Knapsack 0-1 Problem

```c
int valor[maxn], peso[maxn];
int pd[maxc];
int mochila(int n, int c){
	for(int i=0; i<n; ++i)
		for(int j=c; j>=peso[i]; --j)
			pd[j] = max(pd[j], valor[i]+pd[j-peso[i]] );
	return *max_element(pd,pd+c+1);
}
```

##### Backtracking Magic

```c
#include <bits/stdc++.h>
using namespace std;

int n,c;
vector< pair<int,int> > v;
int res,aux;
double c2,aux2;

void bt(int i){
	if(i == n) return;
 
	aux2 = 0; c2 = c;
	for(int j=i; j<n && c2; j++){
		if(v[j].first <= c2){
			c2 -= v[j].first; aux2 += v[j].second;
		} else {
			aux2 += (v[j].second*c2)/v[j].first;
			c2 = 0;
		}
	}
	if(aux2 + aux <= res) return;
  
	if(v[i].first <= c){
		c -= v[i].first;
		aux += v[i].second;
		if(aux > res) res = aux;
		bt(i+1);
		aux -= v[i].second;
		c += v[i].first;
	}
	bt(i+1);
}
 
int main(){
	
	cin>>n>>c;
	for(int i=0; i<n; i++){
		int x,y; cin>>x>>y;
		v.push_backb({x,y});
	}
 
	sort(v.begin(), v.end(), [](pii a, pii b){
		return (a.second+0.0)/a.first > (b.second+0.0)/b.first;
	});
	
	bt(0);
	cout<<res<<endl;
 
	return 0;
}
```

### Longest Increasing Subsequence

Apenas o tamanho da lista

```c
vi lis; int v[SZ];
void LIS(int n){
	lis.clear();
	FOR(i,0,n){
		int j = lowerb(lis, v[i]); // increasing
		//int j = upperb(lis, v[i]); // non-decreasing
		if(j == lis.size()) lis.pb(v[i]);
		else lis[j] = v[i];
}}
```

Imprimir a LIS

```c
vector<pii> aux; vi lis;
int v[SZ],ant[SZ];

void LIS(int n){
	//int h = -1; // increasing
	int h = 1; // non-decreasing
	FOR(i,0,n){
		int j = lowerb( aux, pii(v[i],i*h) );
		if(j == aux.size()) aux.pb({v[i],i*h});
		else aux[j] = {v[i],i*h};
		if(j==0) ant[i] = -1;
		else ant[i] = aux[j-1].se*h;
	}
	lis.clear();
	for(int i = aux.back().se*h; ~i; i = ant[i])
		lis.pb(v[i]);
	reverse(all(lis));
}
```

### Digit DP

```c
int pd[22][200][2]; string dig;

int dp(int i, int k, int f){
	if(pd[i][k][f] != -1) return pd[i][k][f];
	if(i == dig.size()) return test(k) ? 1: 0;
	int res = 0;
	FOR(x,0,10) if(f || x<=dig[i])
		res += dp(i+1, k+x, f||x<dig[i]);
	return pd[i][k][f] = res;
}
 
int query(int k){
	char str[22]; sprintf(str, "%lld", k);
	dig = str; for(auto& c: dig) c -= '0';
	mset(pd,-1); return dp(0,0,0);
}
```

# Matemática

### MDC e MMC

```c
// __gcd(a,b);
int gcd(int a, int b) { return b == 0 ? a : gcd(b, a%b); }
int lcm(int a, int b) { return a*b/gcd(a, b); }
```

### Euclides Extendido

```c
// calculate d = gcd(a,b) and solve ax + by = d
int x,y,d;
void extEucl(int a, int b){
	if(b==0){ x=1; y=0; d=a; return; }
	extEucl(b, a%b);
	int x1 = y;
	int y1 = x - (a/b)*y;
	x = x1; y = y1;
}
```

### Crivo de Eratostenes

```c
// prime check
const int N = 1e6;
bool crivo[N]; vi primos;
for(ll i=2; i<N; ++i){
	if(crivo[i] == true) continue;
	primos.push_back(i);
	for(ll j=i*i; j<N; j+=i)
		crivo[j] = true; }

// divisor count
int count[N];
for(int i=1; i<N; ++i)
	for(int j=i; j<N; j+=i)
		++count[j];

// sum of divisors
int sumdiv[N];
for(int i=1; i<N; ++i)
	for(int j=i; j<N; j+=i)
		sumdiv[j] += i;

// euler's totient
int phi[N];
for(int i=1; i<N; ++i) phi[i] = i;
for(int i=2; i<N; ++i)
	if(phi[i] == i)
		for(int j=i; j<N; j+=i)
			totient[j] -= totient[j]/i;

// biggest prime factor
int factor[N];
for(int i=2; i<N; ++i)
	if(factor[i] == 0)
		for(int j=i; j<N; j+=i)
			factor[j] = i;

// mobius function
int mobius[N]; bool crivo[N];
for(int i=2; i<N; ++i) mobius[i] = -1;
for(int i=2; i<N; ++i)
	if(crivo[i] == false)
		for(int j=i; j<N; j+=i){
			crivo[j] = true;
			mobius[j] *= -1;
			if(j%(i*i)==0) mobius[j] = 0; }
```

### Exponenciação Rápida

```c
const ll mod = 1e9+7;

ll fexp(ll a, ll b){
	ll res = 1; while(b){
		if(b & 1) res = (res*a)%mod;
		a = (a*a)%mod; b >>= 1;
	} return res;
}
```

### Fatoração

```c
map<int,int> fatorar(int n){
	map<int,int> f;
	for(int i=2; i*i <= n; i++)
		while(n%i==0){
			f[i]++; n/=i; }
	if(n > 1) f[n]++;
	return f;
}

//generate all divisors from factors
vector<int> divs;
int aux = 1;
map<int,int> f;
map<int,int>::iterator it;
void bt(){
	if(it == f.end()){ divs.push_back(aux); return; }
	int ant = aux;
	++it; bt(); --it;
	for(int i=0; i<it->second; ++i){
		aux *= it->first;
		++it; bt(); --it;
	} aux = ant;
}
```

### Totiente de Euler

```c
int phi(int n){
	auto f = fatorar(n);
	int res = 1;
	for(auto x: f){
		int fator = x.fi; int exp = x.se;
		res *= fexp(fator,exp-1);
		res *= fator-1;
	}
	return res;
}
```

### Inverso Multiplicativo

```c
/// (a/x) % m = (a * exp(x,-1)) % m
/// exp(x,-1) % m = exp(x, phi(m)-1 ) % m	if gcd(x,m) == 1
/// exp(x,-1) % m = exp(x, m-2) % m		if m is prime
```

### Matrizes

```c
//  F3 = aF2 + bF1 (Recorrencia)
// |a b||F2| = |F3|
// |1 0||F1|   |F2|

vi multx(vi &a, vi &b){
	int n = sqrt(a.size());
	vi res(n*n,0);
	FOR(i,0,n) FOR(j,0,n) FOR(k,0,n)
		res[i*n+j] += a[i*n+k] * b[k*n+j];
	return res;
}

vi powerx(vi &a, int b){
	int n = sqrt(a.size());
	vi res(n*n,0);
	FOR(i,0,n) res[i*n+i] = 1;
	while(b){
		if(b&1) res = multx(res,a);
		a = multx(a,a);
		b = b>>1;
	}
	return res;
}
```

### Miller-Rabin's Prime Check & Pollard Rho's Algorithm

O algoritmo de Rho é usado para fatorar numeros grandes. A função retorna um fator primo P, provavelmente o menor, com complexidade O(sqrt(P)). Para isso, é necessario primeiro verificar se o numero é composto com o teste de primalidade Miller-Rabin.

```c
const int INF = 0x3f3f3f3f;
#define ll long long
ll modSum(ll a, ll b, ll c){
	return (a+b)%c;
}
ll modMul(ll a, ll b, ll c){
	if( a<INF && b<INF ) return (a*b)%c;
	ll res = 0; while(b){
		if(b & 1) res = modSum(res,a,c);
		a = modSum(a,a,c); b >>= 1;
	} return res;
}
ll modExp(ll a, ll b, ll c){
	ll res = 1; while(b){
		if(b & 1) res = modMul(res,a,c);
		a = modMul(a,a,c); b >>= 1;
	} return res;
}

bool rabin(ll n) {
	vector<int> p = {2, 3, 5, 7, 11, 13, 17, 19, 23};
	for(auto x: p) if(n%x==0) return n==x;
	if(n < p.back()) return false;
	ll s = 0, t = n - 1;
	while(~t & 1) t >>= 1, ++s;
	for(auto x: p){
		ll pt = modExp(x, t, n);
		if(pt == 1) continue;
		bool ok = false;
		for(int j = 0; j < s && !ok; j++) {
			if(pt == n - 1) ok = true;
			pt = modMul(pt, pt, n);
		} if(!ok) return false;
	} return true;
}

ll Irand(){ return (((ll)rand())<<15)^rand(); }
ll Lrand(){ return (Irand()<<30)^Irand(); }
ll rho(ll n){
	if(n == 1) return 1;
	if(rabin(n)) return n;
	if(n % 2 == 0) return 2;
	ll d, c = Lrand() % n, x = Lrand() % n, y = x;
	do{ x = modSum(modMul(x, x, n), c, n);
		y = modSum(modMul(y, y, n), c, n);
		y = modSum(modMul(y, y, n), c, n);
		d = __gcd(abs(x - y), n);
	} while(d == 1);
	return d == n ? rho(n) : d;
}
```

### Fast Fourier Transform (FFT)

```c
const double PI = acos(-1);
//typedef complex<double> base;/*
struct base {
	double x, y;
	base() : x(0), y(0) {}
	base(double a, double b=0) : x(a), y(b) {}
	base operator/=(double k) { x/=k; y/=k; return (*this); }
	base operator*(base a) const { return base(x*a.x - y*a.y, x*a.y + y*a.x); }
	base operator*=(base a) {
		double tx = x*a.x - y*a.y;
		double ty = x*a.y + y*a.x;
		x = tx; y = ty;
		return (*this);
	}
	base operator+(base a) const { return base(x+a.x, y+a.y); }
	base operator-(base a) const { return base(x-a.x, y-a.y); }
	double real() { return x; }
	double imag() { return y; }
};//*/
ostream &operator<<(ostream &os, base &p){
	return os<<"("<<round(p.real())<<","<<round(p.imag())<<")";}

void fft (vector<base> & a, bool invert) {
	int n = (int)a.size();
	for(int i=1, j=0; i<n; ++i) {
		int bit = n >> 1;
		for(; j>=bit; bit>>=1)
			j -= bit;
		j += bit;
		if (i < j) swap(a[i], a[j]);
	}

	for(int len=2; len<=n; len<<=1) {
		double ang = 2*PI/len * (invert ? -1 : 1);
		base wlen(cos(ang), sin(ang));
		for(int i=0; i<n; i+=len) {
			base w(1);
			for(int j=0; j<len/2; ++j) {
				base u = a[i+j],  v = a[i+j+len/2] * w;
				a[i+j] = u + v;
				a[i+j+len/2] = u - v;
				w *= wlen;
			}
		}
	}
	if(invert)
		for (int i=0; i<n; ++i)
			a[i] /= n;
}

void convolution(vector<base> a, vector<base> b, vector<base> & res) {
	int n = 1;
	while(n < max(a.size(), b.size())) n <<= 1;
	n <<= 1;
	a.resize(n), b.resize(n);
	fft(a, false); fft(b, false);
	res.resize(n);
	for(int i=0; i<n; ++i) res[i] = a[i]*b[i];
	fft(res, true);
}

int main(){
	
	vector<base> A, B, C;
	A = {base(3),base(2),base(1)};
	B = {base(6),base(5),base(4),base(3)};
	convolution(A, B, C);
	for(int i=0; i<A.size()+B.size()-1; i++) {
		printf(" %f", round(C[i].real()) );
	} printf("\n");
	
	return 0;
}
```

### Congruência Linear

```c
// Ax = B mod M
int A,B,M;
int c = __gcd(A,B,M);
int a = A/c; int b = B/c; int m = M/c;
// solve ax = b mod m
if(__gcd(a,m) != 1) cout<<"No solution"<<endl;
else{
	// x = b/a mod m
	x = (b*fexp(a,phi(m)-1,m))%m;
	cout<<"Solution: "<<x<<endl;
}
```

### Teorema do Resto Chinês

```c
ll fexp(ll a, ll b, ll c){
	ll res = 1; while(b){
		if(b & 1) res = (res*a)%c;
		a = (a*a)%c; b >>= 1;
	} return res;
}

map<int,int> fatorar(int n){
	map<int,int> f;
	for(int i=2; i*i <= n; i++)
		while(n%i==0){
			f[i]++; n/=i; }
	if(n > 1) f[n]++;
	return f;
}

int phi(int n){
	auto f = fatorar(n);
	int res = 1;
	for(auto x: f){
		int fator = x.fi; int exp = x.se;
		res *= round(pow(fator,exp-1));
		res *= fator-1;
	}
	return res;
}

/*

gcd(p[i].se, p[j].se) == 1 (all coprimes)

x = p[0].fi mod p[0].se
x = p[1].fi mod p[1].se
x = p[2].fi mod p[2].se
...
x = p[n].fi mod p[n].se

FIND x

solve:
x = X mod M
where {
	M = p[0].se * p[1].se * p[2].se * ... * p[n].se (product of all p[i].se)
	X = p[0].fi*A[0]*B[0] + p[1].fi*A[1]*B[1] + ... + p[n].fi*A[n]*B[n]
	where {
		A[i] = M/p[i].se (product of all p[j].se where j != i)
		B[i] = A[i]^(-1) mod p[i].se
	}
}

source: pt.wikipedia
*/

void solve(){
	int n = 4;
	pair<int,int> p[n];
	p[0] = {3,5};
	p[1] = {5,13};
	p[2] = {7,29};
	p[3] = {1,41};
	
	ll M = 1;
	for(int i=0; i<n; i++) M *= p[i].se;
	ll A[n];
	for(int i=0; i<n; i++) A[i] = M/p[i].se;
	ll B[n];
	for(int i=0; i<n; i++) B[i] = fexp(A[i], phi(p[i].se)-1, p[i].se );
	ll X = 0;
	for(int i=0; i<n; i++) X += p[i].fi*A[i]*B[i];

	int x = X%M;
	cout<<x<<endl;
}
```

# Geometria

```
Equação da Circunferência (Centro (a,b) e Raio r)
. (x-a)²+(y-b)²=r²

Fórmulas para um triângulo com lados a,b,c
.    Semi-Perímetro: p = (a+b+c)/2
.              Area: A² = p(p-a)(p-b)(p-c)
.              Area: A = bc.sin(alpha)/2
.            Altura: h = 2A/b
.     Raio Inscrito: r = A/p
. Raio Curcunscrito: R = (abc)/(4A)
.              Lado: a² = b²+c²-2bc.cos(alpha)
```

##### Pontos e Linhas

```c
template<class T> bool inOrder(T& a, T& b, T& c){ return a<=b+EPS && b<=c+EPS; }
struct point{
	double x,y;
	point(){}
	point(double a, double b): x(a), y(b) {}
	point operator+(const point &a) const { return point(x+a.x, y+a.y); }
	point operator-(const point &a) const { return point(x-a.x, y-a.y); }
	bool operator == (point p) const {
		return fabs(x-p.x) < EPS && fabs(y-p.y) < EPS; }
	point rotate(double a){ //graus
		a *= PI/180.0; return point( cos(a)*x-sin(a)*y, sin(a)*x+cos(a)*y ); }
	bool isInSegment(point a, point b){
		return (
			(inOrder(a.x,x,b.x) || inOrder(b.x,x,a.x)) &&
			(inOrder(a.y,y,b.y) || inOrder(b.y,y,a.y))
		) && fabs(
			(a.x*b.y + b.x*y + x*a.y) -
			(a.y*b.x + b.y*x + y*a.x)
		) < EPS;
	}
};
struct line{
	double a,b,c;
	line(double x, double y, double z): a(x), b(y), c(z) {}
	line(point A, point B){ a = A.y - B.y; b = B.x - A.x; c = A.x*B.y - A.y*B.x; }
	point somePoint(double k){
		return fabs(b)>EPS ? point(k, -(a*k + c)/b ) : point( -(b*k + c)/a ,k);
	}
};
double dist2(point a, point b){
	return (a.x-b.x)*(a.x-b.x) + (a.y-b.y)*(a.y-b.y);
}
array<double,3> det(line r, line s){
	return {r.b*s.c - r.c*s.b, r.c*s.a - r.a*s.c, r.a*s.b - r.b*s.a};
}
bool isParallel(line r, line s){
	return fabs(det(r,s)[2]) < EPS;
}
point intersection(line r, line s){
	auto res = det(r,s);
	return point(res[0]/res[2], res[1]/res[2]);
}
ostream &operator<<(ostream &os, const point &p){return os<<"("<<p.x<<","<<p.y<<")";}
ostream &operator<<(ostream &os, const line &r){return os<<"["<<r.a<<","<<r.b<<","<<r.c<<"]";}
```

##### Biblioteca Completa

```c
const double EPS = 1e-9;
struct PT {
	double x, y;
	PT() {}
	PT(double x, double y) : x(x), y(y) {}
	PT(const PT &p) : x(p.x), y(p.y) {}
	PT operator + (const PT &p) const { return PT(x+p.x, y+p.y); }
	PT operator - (const PT &p) const { return PT(x-p.x, y-p.y); }
	PT operator * (double c) const { return PT(x*c, y*c ); }
	PT operator / (double c) const { return PT(x/c, y/c ); }
	bool operator == (PT p) const {
		return (fabs(x-p.x) < EPS && (fabs(y-p.y) < EPS)); };
	bool operator < (PT p) const {
		if(fabs(x-p.x) > EPS) return x<p.x; return y<p.y; };
};
// dot(p,q) = length(p)*length(q)*cos(angle between p and q)
double dot(PT p, PT q) { return p.x*q.x+p.y*q.y; }
double dist2(PT p, PT q) { return dot(p-q,p-q); }
double dist(PT p, PT q) { return sqrt(dist2(p,q)); }
double mdist(PT p, PT q) { return fabs(p.x-q.x)+fabs(p.y-q.y); }
double cross(PT p, PT q) { return p.x*q.y-p.y*q.x; }
ostream &operator<<(ostream &os, const PT &p) {return os<<"("<<p.x<<","<<p.y<<")";}
// rotate a point CCW or CW around the origin
PT RotateCCW90(PT p) { return PT(-p.y,p.x); }
PT RotateCW90(PT p) { return PT(p.y,-p.x); }
PT RotateCCW(PT p, double t) {
	return PT(p.x*cos(t)-p.y*sin(t), p.x*sin(t)+p.y*cos(t));
}
// returns angle aob in rad
double angle(PT a, PT o, PT b){
	return acos(dot(a-o,b-o)/sqrt(dot(a-o,a-o)*dot(b-o,b-o)));
}
// returns true if point r is on the left side of line pq
bool ccw(PT p, PT q, PT r) {
	return cross(p,q)+cross(q,r)+cross(r,p) > 0;
}
// project point c onto line through a and b
// assuming a != b
PT ProjectPointLine(PT a, PT b, PT c) {
	return a + (b-a)*dot(c-a, b-a)/dot(b-a, b-a);
}
// project point c onto line segment through a and b
PT ProjectPointSegment(PT a, PT b, PT c) {
	double r = dot(b-a,b-a);
	if (fabs(r) < EPS) return a;
	r = dot(c-a, b-a)/r;
	if (r < 0) return a;
	if (r > 1) return b;
	return a + (b-a)*r;
}
// compute distance from c to segment between a and b
double DistancePointSegment(PT a, PT b, PT c) {
	return sqrt(dist2(c, ProjectPointSegment(a, b, c)));
}
// compute distance between point (x,y,z) and plane ax+by+cz=d
double DistancePointPlane(double x, double y, double z,
double a, double b, double c, double d){
	return fabs(a*x+b*y+c*z-d)/sqrt(a*a+b*b+c*c);
}
// determine if lines from a to b and c to d are parallel or collinear
bool LinesParallel(PT a, PT b, PT c, PT d) {
	return fabs(cross(b-a, c-d)) < EPS;
}
bool LinesCollinear(PT a, PT b, PT c, PT d) {
	return LinesParallel(a, b, c, d)
		&& fabs(cross(a-b, a-c)) < EPS
		&& fabs(cross(c-d, c-a)) < EPS;
}
// determine if line segment from a to b intersects with
// line segment from c to d
bool SegmentsIntersect(PT a, PT b, PT c, PT d) {
	if (LinesCollinear(a, b, c, d)) {
		if (dist2(a, c) < EPS || dist2(a, d) < EPS ||
			dist2(b, c) < EPS || dist2(b, d) < EPS) return true;
		if (dot(c-a, c-b) > 0 && dot(d-a, d-b) > 0 && dot(c-b, d-b) > 0)
			return false;
		return true;
	}
	if (cross(d-a, b-a) * cross(c-a, b-a) > 0) return false;
	if (cross(a-c, d-c) * cross(b-c, d-c) > 0) return false;
	return true;
}
// compute intersection of line passing through a and b
// with line passing through c and d, assuming that unique
// intersection exists; for segment intersection, check if
// segments intersect first
PT ComputeLineIntersection(PT a, PT b, PT c, PT d) {
	b=b-a; d=c-d; c=c-a;
	assert(dot(b, b) > EPS && dot(d, d) > EPS);
	return a + b*cross(c, d)/cross(b, d);
}
// compute center of circle given three points
PT ComputeCircleCenter(PT a, PT b, PT c) {
	b=(a+b)/2;
	c=(a+c)/2;
	return ComputeLineIntersection(b, b+RotateCW90(a-b), c, c+RotateCW90(a-c));
}
// determine if point is in a possibly non-convex polygon (by William
// Randolph Franklin); returns 1 for strictly interior points, 0 for
// strictly exterior points, and 0 or 1 for the remaining points.
// Note that it is possible to convert this into an *exact* test using
// integer arithmetic by taking care of the division appropriately
// (making sure to deal with signs properly) and then by writing exact
// tests for checking point on polygon boundary
bool PointInPolygon(const vector<PT> &p, PT q) {
	bool c = 0;
	for (int i = 0; i < p.size(); i++){
		int j = (i+1)%p.size();
		if ((p[i].y <= q.y && q.y < p[j].y ||
			p[j].y <= q.y && q.y < p[i].y) &&
			q.x < p[i].x + (p[j].x - p[i].x) * (q.y - p[i].y) / (p[j].y - p[i].y))
				c = !c;
	}
	return c;
}
// determine if point is on the boundary of a polygon
bool PointOnPolygon(const vector<PT> &p, PT q) {
	for (int i = 0; i < p.size(); i++)
		if (dist2(ProjectPointSegment(p[i], p[(i+1)%p.size()], q), q) < EPS)
			return true;
	return false;
}
// compute intersection of line through points a and b with
// circle centered at c with radius r > 0
vector<PT> CircleLineIntersection(PT a, PT b, PT c, double r) {
	vector<PT> ret;
	b = b-a;
	a = a-c;
	double A = dot(b, b);
	double B = dot(a, b);
	double C = dot(a, a) - r*r;
	double D = B*B - A*C;
	if (D < -EPS) return ret;
	ret.push_back(c+a+b*(-B+sqrt(D+EPS))/A);
	if (D > EPS)
		ret.push_back(c+a+b*(-B-sqrt(D))/A);
	return ret;
}
// compute intersection of circle centered at a with radius r
// with circle centered at b with radius R
vector<PT> CircleCircleIntersection(PT a, PT b, double r, double R) {
	vector<PT> ret;
	double d = sqrt(dist2(a, b));
	if (d > r+R || d+min(r, R) < max(r, R)) return ret;
	double x = (d*d-R*R+r*r)/(2*d);
	double y = sqrt(r*r-x*x);
	PT v = (b-a)/d;
	ret.push_back(a+v*x + RotateCCW90(v)*y);
	if (y > 0)
		ret.push_back(a+v*x - RotateCCW90(v)*y);
	return ret;
}
// This code computes the area or centroid of a (possibly nonconvex)
// polygon, assuming that the coordinates are listed in a clockwise or
// counterclockwise fashion. Note that the centroid is often known as
// the "center of gravity" or "center of mass".
double ComputeSignedArea(const vector<PT> &p) {
	double area = 0;
	for(int i = 0; i < p.size(); i++) {
		int j = (i+1) % p.size();
		area += p[i].x*p[j].y - p[j].x*p[i].y;
	}
	return area / 2.0;
}
double ComputeArea(const vector<PT> &p) {
	return fabs(ComputeSignedArea(p));
}
// gravity center
PT ComputeCentroid(const vector<PT> &p) {
	PT c(0,0);
	double scale = 6.0 * ComputeSignedArea(p);
	for (int i = 0; i < p.size(); i++){
		int j = (i+1) % p.size();
		c = c + (p[i]+p[j])*(p[i].x*p[j].y - p[j].x*p[i].y);
	}
	return c / scale;
}
// tests whether or not a given polygon (in CW or CCW order) is simple
// segments do not intersect
bool IsSimple(const vector<PT> &p) {
	for (int i = 0; i < p.size(); i++) {
		for (int k = i+1; k < p.size(); k++) {
			int j = (i+1) % p.size();
			int l = (k+1) % p.size();
			if (i == l || j == k) continue;
			if (SegmentsIntersect(p[i], p[j], p[k], p[l]))
				return false;
		}
	}
	return true;
}
```

# Misc

### Codeforces Filter Script

```javascript
for(let e = $("tr")[1]; e; e = e.nextElementSibling)
	if(e.children[1].children[0].alt != "Brazil" && e.children[1].children[0].alt != "Бразилия")
		e.style.display = "none";
```

### Visual Code Settings

```
"window.menuBarVisibility": "toggle",
"workbench.activityBar.visible": true,
"workbench.statusBar.visible": false,
"editor.trimAutoWhitespace": false,
"editor.autoClosingBrackets": "never",
"editor.detectIndentation": false,
"editor.insertSpaces": false,
"editor.acceptSuggestionOnEnter": "off",
"extensions.ignoreRecommendations": true
```

### Bash Script

```
if [ -z $1 ]
then
	echo no arguments
elif [ -z $2 ]
then
	if [ ! -e ${1%.*}.out ] || [ $1 -nt ${1%.*}.out ]
	then
		g++ -std=c++11 -Wfatal-errors $1 -o ${1%.*}.out &&
		echo Compilation Succeeded &&
		./${1%.*}.out
	else
		echo Compilation Succeeded &&
		./${1%.*}.out
	fi
else
	if [ ! -e ${1%.*}.out ] || [ $1 -nt ${1%.*}.out ]
	then
		g++ -std=c++11 -Wfatal-errors $1 -o ${1%.*}.out &&
		echo Compilation Succeeded &&
		./${1%.*}.out < $2
	else
		echo Compilation Succeeded &&
		./${1%.*}.out < $2
	fi
fi
```

### Fast IO

```c
#ifdef _WIN32
#define getchar_unlocked _getchar_nolock
#define putchar_unlocked _putchar_nolock
#endif
int getint(){
	int res = 0;
	int signal = 1;
	bool b = false;
	while(1){
		char c = getchar_unlocked();
		if(c>='0' && c<='9'){
			res = (res<<1) + (res<<3) + (c-'0');
			b = true;
		} else if(c == '-'){
			signal *= -1;
		} else {
			if(b) return res*signal;
		}
	} return -1;
}

char _sint[12]; int _nint;
void putint(int x){
	bool signal = x<0;
	if(signal){
		x = (~x)+1;
		putchar_unlocked('-');
	}
	int ant = x;
	_nint = 0;
	while(ant){
		x = x/10;
		_sint[_nint++] = '0'+ant-x*10;
		ant = x;
	}
	while(_nint--){
		putchar_unlocked(_sint[_nint]);
	}
	putchar_unlocked(' ');
}
```
```
ios_base::sync_with_stdio(0); cin.tie(NULL);
```

### Template

```c
// #include <bits/stdc++.h>/*
#include <algorithm>
#include <iostream>
#include <cstring>
#include <vector>
#include <queue>
#include <stack>
#include <array>
#include <ctime>
#include <cmath>
#include <map>
#include <set>
//*/

using namespace std;

// use cout with pair,array,vector,set,map
template<class T, class W> ostream &operator<<(ostream &os, const pair<T,W> &p){
	return os<<"("<<p.first<<","<<p.second<<")";}
template<class T, size_t W> ostream &operator<<(ostream &os, const array<T,W> &p){
	os<<'(';for(int i=0;i<W;)os<<(i?",":"")<<p[i++]; return os<<')';}
template<class T> ostream &operator<<(ostream &os, const vector<T> &p){
	int b=0;for(auto x:p)os<<(b++?" ":"")<<x; return os;}
template<class T> ostream &operator<<(ostream &os, const set<T> &p){
	int b=0;for(auto x:p)os<<(b++?" ":"")<<x; return os;}
template<class T, class W> ostream &operator<<(ostream &os, const map<T,W> &p){
	int b=0;for(auto x:p)os<<(b++?" ":"")<<x; return os;}

// use print with variadic arguments and static array
template<class T> void print(T X){cout<<X<<endl;}
template <class T, class... W> void print(T X, W... Y){
	cout<<X<<' '; print(Y...); }
template<class T> void print(T* B, T* E){
	while(B!=E) cout<<*(B++)<<' '; cout<<endl; }

#define OR(X,Y) ((X)|(Y))
#define AND(X,Y) ((X)&(Y))
#define XOR(X,Y) ((X)^(Y))
#define NOT(X) (!(X))
#define flip(X) (~(X))
#define shift(X,I) ((I)<0?(1LL*(X))>>(-(I)):(1LL*(X))<<(I))
#define mmax(X,Y) ((X)>(Y)?(X):(Y))
#define mmin(X,Y) ((X)<(Y)?(X):(Y))

#define mset(V,X) memset(V,X,sizeof(V))
#define all(X) (X).begin(),(X).end()
#define upperb(V,X) (int)(upper_bound(all(V),(X))-V.begin())
#define lowerb(V,X) (int)(lower_bound(all(V),(X))-V.begin())
#define FOR(X,L,R) for(int X=L;X<R;++X)

#define int long long
#define ll long long
#define vi vector<int>
#define pii pair<int,int>
#define pb push_back
#define mp make_pair
#define fi first
#define se second
#define endl '\n'

const double EPS = 1e-9, PI = acos(-1);
const ll LINF = 0x3f3f3f3f3f3f3f3f, mod = 1e9+7;
const int INF = 0x3f3f3f3f, SZ = 1e5+20;

int solve(){
	
	return 0;
}

signed main(){
	ios_base::sync_with_stdio(false);cin.tie(NULL);
	
	int t;
	/*
	cin>>t; //*/
	t = 1; //*/
	while(t--) solve();

	return 0;
}
```
