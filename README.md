- [Algoritmos](#algoritmos)
  - [Binary Search](#binary-search)
  - [Mo's Algorithm](#mos-algorithm)
  - [KMP](#kmp)
- [Estruturas de Dados](#estruturas-de-dados)
  - [Fenwick Tree (BIT)](#fenwick-tree-bit)
  - [Union Find](#union-find)
  - [Sparse Table](#sparse-table)
  - [SQRT Decomposition](#sqrt-decomposition)
  - [Ordered Set](#ordered-set)
  - [Segment Tree](#segment-tree)
  - [Trie](#trie)
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
- [Grafos](#grafos)
  - [Dijkstra](#dijkstra)
  - [Spanning Tree (MST)](#spanning-tree-mst)
  - [Lowest Common Ancestor (LCA)](#lowest-common-ancestor-lca)
	- [LCA with Sparse Table](#lca-with-sparse-table)
    - [Binary Lifting & Query on Tree](#binary-lifting--query-on-tree)
  - [Topological Sort](#topological-sort)
  - [SCC Kosaraju](#scc-kosaraju)
  - [Travelling Salesman Problem (TSP)](#travelling-salesman-problem-tsp)
  - [Bipartite Matching (Kuhn's)](#bipartite-matching-kuhns)
- [Programação Dinâmica](#programação-dinâmica)
  - [Knapsack (Mochila)](#knapsack-mochila)
  - [Longest Increasing Subsequence](#longest-increasing-subsequence)
- [Matemática](#matemática)
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
- [Geometria](#geometria)
- [Misc](#misc)
  - [Bash Script](#bash-script)
  - [Visual Code Settings](#visual-code-settings)
  - [Template](#template)

# Algoritmos

### Binary Search

```c
ll bin(ll L, ll R){
	while(L<=R){
		ll m = (L+R)/2;
		test(m) ? R = m-1 : L = m+1;
	}
	return L; // primeiro true
	return R; // ultimo false
}
```

### Mo's Algorithm

```c
int block;
vector<int> v;
vector<array<int,3> > query; // {L,R,index}
int answer[SZ];

void insere(int i){ }
void apaga(int i){ }
int solve(){ }

bool comp(array<int,3> &x, array<int,3> &y){
	if(x[0]/block != y[0]/block)
		return x[0] < y[0];
	return (x[0]/block)&1 ? x[1] > y[1] : x[1] < y[1];
}

void mos(){
	block = sqrt(v.size());
	sort(all(query), comp);
	
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
ll bit[SZ];

ll qry(int idx) {
	ll sum = 0;
	for(int i=idx; i; i -= i&-i) sum += bit[i];
	return sum;
}
void upd(int idx, ll k){
	for(int i=idx; i<SZ; i += i&-i) bit[i] += k;
}
```

### Union Find

```c
int pai[SZ]; // inicializar com pai[i] = i;

int find(int i){ return pai[i]==i ? i : pai[i] = find(pai[i]); }
void uni(int i, int j){ pai[find(i)] = find(j); }
```

### Sparse Table

```c
vector<int> arr;
int st[SZ][20];

void build(){
	int n = arr.size(); int m = log2(n-1)+1;
	FOR(i,0,n) st[i][0] = arr[i];   
	for(int j=1,p=2; j<m; j++,p*=2) FOR(i,0,n-p+1)
		st[i][j] = min( st[i][j-1], st[i+p/2][j-1] );
}

int query(int i, int j){
	if(i==j) return st[i][0];
	if(i>j) swap(i,j);
	int k = log2(j-i);
	return min(st[i][k], st[j+1-(1<<k)][k]);
}
```

### SQRT Decomposition

```c
int n,raiz;
ll v[100010];
ll bl[325];

void build(){
	raiz = sqrt(n);
	FOR(i,0,n) bl[i/raiz] += v[i];
}
ll query(int l, int r, int k){
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
void pointupdate(int i, ll k){
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
//#define SZ 1050000
//#define var int
// MODO DE USAR: build(1,1,n); query(l,r); update(i,x);

int n;
var seg[SZ]; // 1-rooted
var arr[SZ]; // 1-based
var zero;

var mix(var u, var v){
	if(u==zero) return v;
	if(v==zero) return u;
	return u+v;
}

void build(int i, int a, int z){
	if(a==z){
		seg[i] = arr[a];
		return;
	}
	build(i*2, a, (a+z)/2);
	build(i*2+1, (a+z)/2+1, z);
	seg[i] = mix(seg[i*2], seg[i*2+1]);
}

var qry(int i, int a, int z, int l, int r){
	if(l<=a && z<=r) return seg[i];
	if(l>z || r<a) return zero;
	var u = qry(i*2, a, (a+z)/2, l, r);
	var v = qry(i*2+1, (a+z)/2+1, z, l, r);
	return mix(u, v);
}

var query(int l, int r){ return qry(1,1,n,l,r); }

void pointupdate(int index, var x){
	int i=1, l=1, r=n;
	while(l!=r){
		if((l+r)/2 < index) l = (l+r)/2+1, i = i*2+1;
		else r = (l+r)/2, i = i*2;
	}
	arr[index] = x; seg[i] = x; i/=2;
	while(i) seg[i] = mix(seg[i*2], seg[i*2+1]), i/=2;
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

### Suffix Array & Longest Common Prefix Array

```c
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
int a[SZ];

struct wavelet_tree{
	int lo, hi;
	wavelet_tree *l, *r;
	vi b;

	//nos are in range [x,y]
	//array indices are [from, to)
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

	//kth smallest element in [l, r]
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

int main(){
	
	int i,n,k,j,q,l,r,op;
	cin >> n;
	for(int i=0; i<n; i++) cin >> a[i+1];
	wavelet_tree T(a+1, a+n+1, 1, MAX);
	
	cin>>q;
	while(q--){
		cin>>op;
		if(op == 0){
			cerr<<" Kth smallest in [l,r]"<<endl;
			cin >> l >> r >> k;
			cout << T.kth(l, r, k) << endl;
		}
		if(op == 1){
			cerr<<" How many x <= K in [l,r]"<<endl;
			cin >> l >> r >> k;
			cout << T.LTE(l, r, k) << endl;
		}
		if(op == 2){
			cerr<<" How many x == K in [l,r]"<<endl;
			cin >> l >> r >> k;
			cout << T.count(l, r, k) << endl;
		}
	}
	
	return 0;
} 
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
ll bit1[SZ], bit2[SZ];

// retorna o somatorio a[1]+a[2]+...+a[i]
ll query(int idx){
	ll sum1=0, sum2=0;
	for(int i = idx; i; i -= i&-i) sum1 += bit1[i];
	for(int i = idx; i; i -= i&-i) sum2 += bit2[i];
	return sum1*idx - sum2;
}

// incrementa x em todos a[i],a[i+1],...,a[N]
void update(int idx, ll x){
	for(int i = idx; i<SZ; i += i&-i) bit1[i] += x;
	for(int i = idx; i<SZ; i += i&-i) bit2[i] += x*(idx-1);
}
```

##### BIT 2D

```c
int bit[SZ][SZ];

int query(int idx, int jdx){
	int sum = 0;
	for(int i=idx; i; i -= i&-i)
		for(int j=jdx; j; j -= j&-j)
			sum += bit[i][j];
	return sum;
}
void update(int idx, int jdx, ll k){
	for(int i=idx; i<SZ; i += i&-i)
		for(int j=jdx; j<SZ; j += j&-j)
			bit[i][j] += k;
}
// para point update range query
var rangequery(int x1, int y1, int x2, int y2){
	return query(x2,y2) - query(x2,y1-1) - query(x1-1,y2) + query(x1-1,y1-1);
}
// para range update point query
void rangeupdate(int x1, int y1, int x2, int y2, var k){
	update(x1,y1,k); update(x2+1,y1,-k); update(x1,y2+1,-k); update(x2+1,y2+1,k);
}
```

##### BIT 2D Range Update Range Query

```c
ll bit[4][SZ][SZ];

void upd(int id, int x, int y, ll k){
	for(int i = x; i<SZ; i += i&-i)
		for(int j = y; j<SZ; j += j&-j)
			bit[id][i][j] += k;
}

ll qry(int id, int x, int y){
	ll sum = 0;
	for(int i = x; i>0; i -= i&-i)
		for(int j = y; j>0; j -= j&-j)
			sum += bit[id][i][j];
	return sum;
}

void update(int x, int y, ll k){
	upd(0, x, y, k);
	upd(1, x, y, k*(x-1) );
	upd(2, x, y, k*(y-1) );
	upd(3, x, y, k*(x-1)*(y-1) );
}

ll query(int x, int y){
	ll sum = 0;
	sum += qry(0,x,y)*x*y;
	sum -= qry(1,x,y)*y;
	sum -= qry(2,x,y)*x;
	sum += qry(3,x,y);
	return sum;
}

ll rangequery(int x1, int y1, int x2, int y2){
	return query(x2,y2) - query(x2,y1-1) - query(x1-1,y2) + query(x1-1,y1-1);
}

void rangeupdate(int x1, int y1, int x2, int y2, ll k){
	update(x1,y1,k); update(x2+1,y1,-k); update(x1,y2+1,-k); update(x2+1,y2+1,k);
}
```

##### Ordered Multiset with BIT

```c
const int N = 1e6+10, LOGN = 20;
struct ordered_multiset{
	
	int bit[N], contador;
	ordered_multiset(){ memset(bit,0,sizeof bit); contador = 0; }
	
	int size(){
		return contador;
	}
	void insert(int x){ // must be inside [1,N-1]
		for(int i=x; i<N; i += i&-i) bit[i]++; contador++;
	}
	void erase(int x){ // set must contain x
		for(int i=x; i<N; i += i&-i) bit[i]--; contador--;
	}
	int lte(int x){ // how many less than or equal
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
		return pos + 1;
	}
};
```

### Gambiarras

##### Ordered Multiset

```c
#define var pair<int,int>
#define ordered_set tree<var,null_type,less<var>,rb_tree_tag,tree_order_statistics_node_update>

int id = 0; map<int,vi> ids;
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
int bs[1000000000/32];
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
struct ivi{ // inversed vector<int>
	vi a;
	ivi operator=(vi v) { a = v; reverse(all(a)); return (*this); }
	int& operator[](int i){ return a[a.size()-1-i]; }
	void push_front(int x){ a.push_back(x); }
	int size(){ return (int)a.size(); }
	void swap(ivi& b){ a.swap(b.a); }
};
```

# Grafos

### Dijkstra

```c
vector<pii> g[SZ];
int d[SZ];

void dijkstra(int ori){
	memset(d,63,sizeof d);
	d[ori] = 0;

	typedef array<int,2> vet;
	priority_queue< vet, vector<vet>, greater<vet> > pq;
	pq.push( {0,ori} );
	while(!pq.empty()){
		int dv = pq.top()[0];
		int v = pq.top()[1];
		pq.pop();
		if(dv > d[v]) continue;
		for(auto x: g[v]){
			int u = x.fi;
			int p = x.se;
			if(dv + p < d[u]){
				d[u] = dv + p;
				pq.push( {d[u], u} );
			}
		}
	}
}
```

### Spanning Tree (MST)

```c
vector<pii> g[SZ],h[SZ];
	
void prim(){
	priority_queue< array<int,3> > pq;
	for(auto p: g[1]) pq.push( {-p.se, p.fi, 1} );
	bool vis[SZ]; mset(vis,0); vis[1] = 1;
	while(!pq.empty()){
		int v = pq.top()[1], u = pq.top()[2];
		int d = -pq.top()[0]; pq.pop();
		if(vis[v]) continue; else vis[v]=1;
		h[v].pb({u,d}); h[u].pb({v,d});
		for(auto& p: g[v]) pq.push({-p.se, p.fi, v});
	}
}
```

### Lowest Common Ancestor (LCA)

##### LCA with Sparse Table

```c
vi g[SZ];
int cont;
int id[SZ]; // ultimo contador do vertice
vector<pii> arr; // array pro lca

void linear(int v, int ant, int altura){
	id[v] = cont++;
	arr.pb({altura,v});
	for(auto u: g[v]) if(u!=ant){
		linear(u, v, altura+1);
		id[v] = cont++;
		arr.pb({altura,v});
	}
}

pii st[SZ*2][21]; // must be [n][logn+1]
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

```c
int grau[SZ]; int n;
vi g[SZ]; vi res;

bool ord_top(){ // kahn's algorithm
	priority_queue<int> pq; res.clear();
	bool vis[SZ]; memset(vis,1,sizeof vis);
	for(int i=0; i<n; i++) if(grau[i]==0){
		vis[i] = 0;
		pq.push(-i);
	}
	while(!pq.empty()){
		int v = -pq.top(); pq.pop();
		res.pb(v);
		for(int i=0; i<g[v].size(); i++){
			int u = g[v][i];
			if(vis[u]){
				grau[u]--;
				if(grau[u] == 0){
					vis[u] = 0;
					pq.push(-u);
	}}}}
	return res.size() == n;
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
double h[MAXN][MAXN];
double tsp[1<<MAXN][MAXN];
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

# Programação Dinâmica

### Knapsack (Mochila)

```c
int n,c;
int valor[1010];
int volume[1010];
int pd[1010][1010];

void knapsack(){
	mset(pd,0);
	FOR(j,1,c+1) if(volume[0] <= j) pd[j][0] = valor[0];
	FOR(i,1,n) FOR(j,1,c+1){
		pd[j][i] = max(pd[j-1][i], pd[j][i-1]);
		if(volume[i] <= j) pd[j][i] = max(pd[j][i], valor[i]+pd[j-volume[i]][i-1]);
	}
}
```


### Longest Increasing Subsequence

Apenas o tamanho da lista

```c
vi lis;
int n;
scanf("%d", &n);
int v[n];
for(int i=0; i<n; i++){
	scanf("%d", &v[i]);
	int j = lower_bound(all(lis), v[i]) - lis.begin();
	if(j==lis.size()) lis.pb(v[i]);
	else lis[j] = v[i];
}
printf("%d\n", (int)lis.size());
```

Imprimir a LIS

```c
vector<pii> lis;
int n;
scanf("%d", &n);
int v[n],ant[n];
for(int i=0; i<n; i++){
	scanf("%d", &v[i]);
	int j = lower_bound( all(lis), pii(v[i],-i) ) - lis.begin();
	if(j==lis.size()) lis.pb({v[i],-i});
	else lis[j] = {v[i],-i};
	if(j==0) ant[i] = -1;
	else ant[i] = -lis[j-1].se;
}
deque<int> res;
int index = -lis.back().se;
do{
	res.push_front(v[index]);
	index = ant[index];
} while(index != -1);

printf("Tamanho da LIS: %d\n", (int)res.size());
printf("LIS: ");
for(auto i: res) printf("%d ", i); printf("\n");
```

# Matemática

### MDC e MMC

```c
// __gcd(a,b);
int mdc(int a, int b) { return b == 0 ? a : mdc(b, a%b); }
int mmc(int a, int b) { return a*b/mdc(a, b); }
```

### Euclides Extendido

```c
int x,y,d;
// x*a + y*b = d
// d = mdc(a,b)
// x*a = d mod b

void extEucl(int a, int b){
	if(b==0){
		x=1; y=0; d=a; return;
	}
	extEucl(b, a%b);
	int x1 = y;
	int y1 = x - (a/b)*y;
	x = x1; y = y1;
}
```

### Crivo de Eratostenes

```c
const int N = 1e6;
bool crivo[N]; vi primes;
void build(){
	for(int i=2; i<N; i++){
		if(crivo[i]==0){ primes.pb(i);
			for(int j=i*2; j<N; j+=i){
				crivo[j]=1;
}}}}
```

### Exponenciação Rápida

```c
ll soma(ll a, ll b, ll c=mod){
	return (a+b<c ? a+b : (a+b<2*c ? a+b-c : (a+b)%c) );
}
ll mult(ll a, ll b, ll c=mod){
	ll res = 0;
	while(b){
		if(b&1) res = soma(res, a, c);
		a = soma(a, a, c);
		b /= 2;}
	return res;
}
ll fexp(ll a, ll b, ll c=mod){
	ll res = 1;
	while(b){
		if(b&1) res = mult(res, a, c);
		a = mult(a, a, c);
		b /= 2;}
	return res;
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
```

### Totiente de Euler

```c
var phi(var n){
	auto f = fatorar(n);
	var res = 1;
	for(auto x: f){
		var fator = x.fi; var exp = x.se;
		res *= fexp(fator,exp-1);
		res *= fator-1;
	}
	return res;
}
```

### Inverso Multiplicativo

```c
/// TIP /// inv(x) = x^(m-2) mod m ??? if m is prime and x<m
var inv(var x, var mod){
	if(__gcd(x,mod)!=1) return -1
	var _phi = phi(mod) - 1;
	return fexp(x,_phi,mod);
}
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

O algoritmo de Rho é usado para fatorar numeros grandes. A função retorna o um fator primo P, provavelmente o menor, com complexidade O(sqrt(P)). Para isso, é necessario verificar se o numero é composto com o teste de primalidade Miller-Rabin.

```c
ll modSum(ll a, ll b, ll c){
	return a+b >= c ? (a+b)%c : a+b;
}
ll modMul(ll a, ll b, ll c){
	ll res = 0; while(b){
		if(b & 1) res = modSum(res, a, c);
		a = modSum(a, a, c); b /= 2;
	} return res;
}
ll modExp(ll a, ll b, ll c){
	ll res = 1; while(b){
		if(b & 1) res = modMul(res, a, c);
		a = modMul(a, a, c); b /= 2;
	} return res;
}

bool rabin(ll n) {
	vector<int> p = {2, 3, 5, 7, 11, 13, 17, 19, 23};
	for(auto x: p) if(n%x==0) return n==x;
	if(n < p.back()) return false;
	ll s = 0, t = n - 1;
	while(~t & 1) t >>= 1, ++s;
	for(auto x: p){
		ll pt = modExp((ll)x, t, n);
		if(pt == 1) continue;
		bool ok = false;
		for(int j = 0; j < s && !ok; j++) {
			if(pt == n - 1) ok = true;
			pt = modMul(pt, pt, n);
		} if(!ok) return false;
	} return true;
}

ll Lrand(){ ll tmp=rand(); return (tmp<<31)|rand(); }
ll rho(ll n){
	if(rabin(n)) return n;
	if(n % 2 == 0) return 2;
	ll d, c = Lrand() % n, x = Lrand() % n, y = x;
	do{ x = modSum(modMul(x, x, n), c, n);
		y = modSum(modMul(y, y, n), c, n);
		y = modSum(modMul(y, y, n), c, n);
		d = __gcd(abs(x - y), n);
	} while(d == 1);
	return d;
}
```

### Fast Fourier Transform (FFT)

```c
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
}; //*/
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
		double ang = 2*M_PI/len * (invert ? -1 : 1);
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
		printf(" %.0f", C[i].real());
	} printf("\n");
	
	return 0;
}
```

# Geometria

```
Equação da Circunferência (Centro (a,b) e Raio r)
. (x-a)²+(y-b)²=r²

Fórmulas para um triângulo com lados a,b,c
.    Semi-Perímetro => p = (a+b+c)/2
.              Area => A = sqrt(p(p-a)(p-b)(p-c))
.              Area => A = bc.sin(alpha)/2
.            Altura => h = 2A/b
.     Raio Inscrito => r = A/p
. Raio Curcunscrito => R = (abc)/(4A)
```

##### Biblioteca Completa

```c
const double EPS = 1e-9;
const double PI = acos(-1);
struct PT {
	double x, y;
	PT(){ x = y = 0.0; }
	PT(double _x, double _y) : x(_x), y(_y) {}
	PT operator - (PT other) const {return PT(x-other.x, y-other.y);}
	PT operator + (PT other) const {return PT(x+other.x, y+other.y);}
	PT operator * (double c) const { return PT(x*c, y*c ); }
	PT operator / (double c) const { return PT(x/c, y/c ); }
	bool operator == (PT other) const {
		return fabs(x - other.x) < EPS && fabs(y - other.y) < EPS; }
	bool operator < (PT other) const {
		if(fabs(x - other.x) > EPS) return x < other.x; return y < other.y; }
	
	double dist(PT other){
		return hypot(x - other.x, y - other.y);
	}
	bool isBetween(PT p, PT q){ //maior ou igual
		return fabs(this->dist(p)+this->dist(q)-p.dist(q)) < EPS;
	}
	PT rotate(double rad, PT ori = PT(0.0,0.0)) { //CCW
		PT p = *this - ori;
		return PT(p.x*cos(rad)-p.y*sin(rad), p.x*sin(rad)+p.y*cos(rad)) + ori;
	}
	PT rotate90(int n=1, PT ori = PT(0.0,0.0)) { //CCW
		PT p = PT(x,y)-ori; n = n%4; if(n<0) n+=4;
		while(n--) p = PT(-p.y,p.x);
		return p+ori;
	}
	bool domina(PT other){
		return (fabs(x-other.x) < EPS || x > other.x) && (fabs(y-other.y) < EPS || y > other.y);
	}
};
double dot(PT p, PT q){ return p.x*q.x+p.y*q.y; }
double cross(PT p, PT q){ return p.x*q.y-p.y*q.x; }
double dist2(PT p, PT q){ return dot(p-q,p-q); }
ostream &operator<<(ostream &os, const PT &p){ return os<<"("<<p.x<<","<<p.y<<")"; }

PT ProjectPointLine(PT a, PT b, PT c) { // a != b
	return a + (b-a)*dot(c-a, b-a)/dot(b-a, b-a);
}
PT ProjectPointSegment(PT a, PT b, PT c) {
	double r = dot(b-a,b-a);
	if (fabs(r) < EPS) return a;
	r = dot(c-a, b-a)/r;
	if (r < 0) return a;
	if (r > 1) return b;
	return a + (b-a)*r;
}
double DistancePointSegment(PT a, PT b, PT c) {
	return sqrt(dist2(c, ProjectPointSegment(a, b, c)));
}
double DistancePointPlane(double x, double y, double z, double a, double b, double c, double d){
	return fabs(a*x+b*y+c*z-d)/sqrt(a*a+b*b+c*c);
}
bool LinesParallel(PT a, PT b, PT c, PT d) {
	return fabs(cross(b-a, c-d)) < EPS;
}
bool LinesCollinear(PT a, PT b, PT c, PT d) {
	return LinesParallel(a, b, c, d)
		&& fabs(cross(a-b, a-c)) < EPS
		&& fabs(cross(c-d, c-a)) < EPS;
}
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
PT ComputeLineIntersection(PT a, PT b, PT c, PT d) {
	b=b-a; d=c-d; c=c-a;
	assert(dot(b, b) > EPS && dot(d, d) > EPS);
	return a + b*cross(c, d)/cross(b, d);
}
PT ComputeCircleCenter(PT a, PT b, PT c) {
	b=(a+b)/2; c=(a+c)/2;
	return ComputeLineIntersection(b, b+(a-b).rotate90(-1), c, c+(a-c).rotate90(-1));
}
bool PointInPolygon(const vector<PT> &p, PT q) {
	bool c = 0;
	for (int i = 0; i < p.size(); i++){
		int j = (i+1)%p.size();
		if ((p[i].y <= q.y && q.y < p[j].y || p[j].y <= q.y && q.y < p[i].y) &&
			q.x < p[i].x + (p[j].x - p[i].x) * (q.y - p[i].y) / (p[j].y - p[i].y))
				c = !c;
	}
	return c;
}
bool PointOnPolygon(const vector<PT> &p, PT q) {
	for (int i = 0; i < p.size(); i++)
		if (dist2(ProjectPointSegment(p[i], p[(i+1)%p.size()], q), q) < EPS)
			return true;
	return false;
}
vector<PT> CircleLineIntersection(PT a, PT b, PT c, double r) {
	vector<PT> ret; b = b-a; a = a-c;
	double A = dot(b, b); double B = dot(a, b);
	double C = dot(a, a) - r*r; double D = B*B - A*C;
	if (D < -EPS) return ret;
	ret.push_back(c+a+b*(-B+sqrt(D+EPS))/A);
	if (D > EPS) ret.push_back(c+a+b*(-B-sqrt(D))/A);
	return ret;
}
vector<PT> CircleCircleIntersection(PT a, PT b, double r, double R) {
	vector<PT> ret; double d = sqrt(dist2(a, b));
	if (d > r+R || d+min(r, R) < max(r, R)) return ret;
	double x = (d*d-R*R+r*r)/(2*d);
	double y = sqrt(r*r-x*x);
	PT v = (b-a)/d;
	ret.push_back(a+v*x + v.rotate90()*y);
	if (y > 0) ret.push_back(a+v*x - v.rotate90()*y);
	return ret;
}
double ComputeSignedArea(const vector<PT> &p) {
	double area = 0;
	for(int i = 0; i < p.size(); i++) {
		int j = (i+1) % p.size();
		area += p[i].x*p[j].y - p[j].x*p[i].y;
	} return area / 2.0;
}
double ComputeArea(const vector<PT> &p) {
	return fabs(ComputeSignedArea(p));
}
PT ComputeCentroid(const vector<PT> &p) { //centro de gravidade
	PT c(0,0);
	double scale = 6.0 * ComputeSignedArea(p);
	for (int i = 0; i < p.size(); i++){
		int j = (i+1) % p.size();
		c = c + (p[i]+p[j])*(p[i].x*p[j].y - p[j].x*p[i].y);
	} return c / scale;
}
bool IsSimple(const vector<PT> &p) { //linhas nao se intersectam
	for (int i = 0; i < p.size(); i++) {
		for (int k = i+1; k < p.size(); k++) {
			int j = (i+1) % p.size();
			int l = (k+1) % p.size();
			if (i == l || j == k) continue;
			if (SegmentsIntersect(p[i], p[j], p[k], p[l]))
				return false;
	}} return true;
}
```

# Misc

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
		echo go &&
		./${1%.*}.out
	else
		echo go &&
		./${1%.*}.out
	fi
else
	if [ ! -e ${1%.*}.out ] || [ $1 -nt ${1%.*}.out ]
	then
		g++ -std=c++11 -Wfatal-errors $1 -o ${1%.*}.out &&
		echo go &&
		./${1%.*}.out < $2
	else
		echo go &&
		./${1%.*}.out < $2
	fi
fi
```

### Visual Code Settings

```
"editor.trimAutoWhitespace": false,
"editor.autoClosingBrackets": false,
"editor.detectIndentation": false,
"editor.insertSpaces": false,
"editor.acceptSuggestionOnEnter": "off",
"extensions.ignoreRecommendations": true
```

### Template

```c
//#include <bits/stdc++.h>/*
#include <functional> //greater
#include <algorithm> //sort
#include <iostream>
#include <iterator>
#include <utility> //pair
#include <sstream>
#include <cstring> //memset
#include <complex>
#include <cassert>
#include <cstdio>
#include <string>
#include <vector>
#include <cctype> //isaplha, tolower
#include <deque>
#include <queue>
#include <stack>
#include <array>
#include <cmath> //sqrt, sin
#include <ctime>
#include <map>
#include <set>
//*/

#define ll long long
#define vi vector<int>
#define pii pair<int,int>
#define pb push_back
#define mp make_pair
#define fi first
#define se second
#define endl '\n'
using namespace std;

template<class T> ostream &operator<<(ostream &os, const pair<T,T> &p){
	return os<<"("<<p.fi<<","<<p.se<<")";}
template<class T> ostream &operator<<(ostream &os, const pair<const T,T> &p){
	return os<<"("<<p.fi<<","<<p.se<<")";}
template<class T> ostream &operator<<(ostream &os, const array<T,3> &p){
	return os<<"("<<p[0]<<","<<p[1]<<","<<p[2]<<")";}
template<class T> ostream &operator<<(ostream &os, const vector<T> &p){
	for(auto x: p) os<<x<<' '; return os;}
#define FOR(X,L,R) for(int X=L;X<R;X++)
#define FIR(X,R,L) for(int X=R;X>=L;X--)
#define FI(I,X) for(auto I=(X).begin();I!=(X).end();I++)
#define print(X) {cerr<<"{ ";FI(I,X)cerr<<*I<<' ';cerr<<"}"<<endl;}
#define mset(V,X) memset(V,X,sizeof(V))
#define all(X) (X).begin(),(X).end()
#define upperb(V,X) (int)(upper_bound(all(V),(X))-V.begin())
#define lowerb(V,X) (int)(lower_bound(all(V),(X))-V.begin())

const double EPS = 1e-9, PI = acos(-1);
const ll LINF = 0x3f3f3f3f3f3f3f3f, mod = 1e9+7;
const int INF = 0x3f3f3f3f, SZ = 1e5+20;

int main(){
	
	return 0;
}
```
