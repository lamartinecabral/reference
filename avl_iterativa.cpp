//#include <bits/stdc++.h>/*
#include <iostream>
#include <vector>
#include <ctime>
#include <cmath>
#include <set>
//*/

using namespace std;

class no{
  public:
  int k,v,l,r,p,s,h;
  no(){ k = v = l = r = p = 0; s = h = 1; }
  no(int key, int value){
    k = key; v = value;
    l = r = p = 0; s = h = 1;
  }
};

class avl{

  public:
  int root;
  vector<no> tree;
  // set<int> freeIndexes;
  vector<int> freeIndexes;
  avl(){ tree = {no()}; root = 0; }

  int newnode(){
    if(freeIndexes.size()){
      // int ret = *freeIndexes.begin(); freeIndexes.erase(freeIndexes.begin()); return ret;
      int ret = freeIndexes.back(); freeIndexes.pop_back(); return ret;
    }
    tree.push_back(no());
    return tree.size() - 1;
  }
  void free(int node) {
    tree[node] = no();
    // freeIndexes.insert(node);
    freeIndexes.push_back(node);
  }

  int find(int key){
    int node = root;
    while(node && tree[node].k != key)
      node = key < tree[node].k ? tree[node].l : tree[node].r;
    return node;
  }

  int smaller(int node){
    while (tree[node].l)
      node = tree[node].l;
    return node;
  }

  int greater(int node){
    while (tree[node].r)
      node = tree[node].r;
    return node;
  }

  int next(int node) {
    if(tree[node].r)
      return smaller(tree[node].r);
    while(tree[node].p && node == tree[tree[node].p].r)
      node = tree[node].p;
    if(tree[node].p)
      return tree[node].p;
    return 0;
  }

  int prev(int node) {
    if(tree[node].l)
      return greater(tree[node].l);
    while(tree[node].p && node == tree[tree[node].p].l)
      node = tree[node].p;
    if(tree[node].p)
      return tree[node].p;
    return 0;
  }

  int kth(int i){
    int node = root;
    while(node) {
      if (i > tree[node].s)
        return 0;
      int esq = (!tree[node].l ? 1 : tree[tree[node].l].s + 1);
      if (i == esq)
        return node;
      if (i > esq) {
        i -= esq;
        node = tree[node].r;
      }
      else {
        node = tree[node].l;
      }
    }
    return 0;
  }

  int order(int key) {
    int node = root;
    int i = 0;
    while(node) {
      int esq = 1 + (tree[node].l ? tree[tree[node].l].s : 0);
      if (key == tree[node].k) {
        return i + esq;
      }
      else if (key > tree[node].k) {
        i += esq;
        node = tree[node].r;
      }
      else {
        node = tree[node].l;
      }
    }
    return i; 
  }

  void updateSize(int node, bool toRoot = false){
    do {
      tree[node].s =
        (tree[node].l ? tree[tree[node].l].s : 0) +
        (tree[node].r ? tree[tree[node].r].s : 0) + 1;
      tree[node].h = 1 + max(
        (tree[node].l ? tree[tree[node].l].h : 0),
        (tree[node].r ? tree[tree[node].r].h : 0)
      );
    } while (toRoot && (node = tree[node].p));
  };

  void connect(int parent, int child, int cchild = 0) {
    if(!cchild)
      cchild = child;
    if(parent){
      if(tree[parent].k > tree[cchild].k)
        tree[parent].l = child;
      else
        tree[parent].r = child;
    }
    else
      root = child;
    if(child)
      tree[child].p = parent;
  }

  void rotate(int X, int Z, int Y) {
    connect(tree[X].p, Z); connect(Z, X); connect(X, Y, Z);
  }
  void rotateR(int node, bool update = true){
    int D = node;
    int B = tree[D].l;
    int C = tree[B].r;
    rotate(D, B, C);
    if(update)
      updateSize(D, true);
  }
  void rotateL(int node, bool update = true){
    int D = node;
    int F = tree[D].r;
    int E = tree[F].l;
    rotate(D, F, E);
    if(update)
      updateSize(D, true);
  }

  void balance(int node){
    while(node){
      updateSize(node);
      int lheight = (tree[node].l ? tree[tree[node].l].h : 0);
      int rheight = (tree[node].r ? tree[tree[node].r].h : 0);
      
      if(abs(lheight - rheight) <= 1) {
        node = tree[node].p;
        continue;
      }
      
      if(lheight < rheight){
        int E = tree[tree[node].r].l;
        if(E && tree[E].h == tree[node].h - 2) {
          int upd = tree[node].r;
          rotateR(tree[node].r, false);
          updateSize(upd);
          updateSize(tree[node].r);
        }
      } else {
        int E = tree[tree[node].l].r;
        if(E && tree[E].h == tree[node].h - 2) {
          int upd = tree[node].l;
          rotateL(tree[node].l, false);
          updateSize(upd);
          updateSize(tree[node].l);
        }
      }
      
      if (lheight < rheight)
        rotateL(node, false);
      else
        rotateR(node, false);
      
      updateSize(node);
      node = tree[node].p;
    }
  }

  int add(int k, int v = 0) {
    int novo = newnode();
    tree[novo] = no(k,v);
    if (!root)
      return root = novo;
    int node = root;
    while(true){
      if(tree[node].k == tree[novo].k)
        return free(novo), 0;
      if(tree[node].k > tree[novo].k){
        if(tree[node].l)
          node = tree[node].l;
        else break;
      } else {
        if(tree[node].r)
          node = tree[node].r;
        else break;
      }
    }
    connect(node, novo);
    balance(node);
    return novo;
  }

  bool remove(int node){
    int node2 = smaller(tree[node].r);
    if(!node2)
      node2 = greater(tree[node].l);
    if(!node2) {
      int parent = tree[node].p;
      if(!parent)
        root = 0;
      else {
        connect(parent, 0, node);
        updateSize(parent, true);
      }
      free(node);
      return true;
    }
    if(tree[node2].l)
      connect(tree[node2].p, tree[node2].l, node2);
    else
      connect(tree[node2].p, tree[node2].r, node2);
    swap(tree[node].k, tree[node2].k);
    swap(tree[node].v, tree[node2].v);
    updateSize(tree[node2].p, true);
    free(node2);
    return true;
  }

  int height(){
    if(!root) return 0;
    return tree[root].h;
  }

  // string checkConsistence(){
  //   if(!root)
  //     return "no nodes";
  //   if(tree[root].p)
  //     return "root has parent";
  //   for(int node = 1; node < tree.size(); node++) {
  //     if(freeIndexes.count(node))
  //       continue;
  //     if(!tree[node].p) {
  //       if(node != root)
  //         return "node without parent";
  //     }
  //     else {
  //       if (tree[tree[node].p].k > tree[node].k) {
  //         if(tree[tree[node].p].l != node)
  //           return "pointer error";
  //       }
  //       else {
  //         if(tree[tree[node].p].r != node)
  //           return "pointer error";
  //       }
  //     }
  //     if(tree[node].l)
  //       if(tree[tree[node].l].k >= tree[node].k)
  //         return "left child key is not less";
  //     if(tree[node].r)
  //       if(tree[tree[node].r].k <= tree[node].k)
  //         return "right child key is not greater";
  //     int lsize = (tree[node].l ? tree[tree[node].l].s : 0);
  //     int rsize = (tree[node].r ? tree[tree[node].r].s : 0);
  //     if (tree[node].s != 1 + lsize + rsize)
  //       return "wrong size";
  //     int lheight = (tree[node].l ? tree[tree[node].l].h : 0);
  //     int rheight = (tree[node].r ? tree[tree[node].r].h : 0);
  //     if(tree[node].h != max(lheight, rheight) + 1)
  //       return "wrong height";
  //   }
  //   return "";
  // }

};

signed main(){
  
  srand(time(0));

  int n = 2e5;
  //*
  auto t = avl();
  string err = "";
  for(int i=1; i<=n; i++){
    int key = 1+( rand()%(n*10) );
    int node = t.find(key);
    if(node) t.remove(node);
    else t.add(key);
    // if(i%(n/10) == 0){
    //   err = t.checkConsistence();
    //   if(err != ""){
    //     cout << i << ' ' << err << endl;;
    //     break;
    //   }
    // }
  }
  if(err == "") cout << t.height() << endl;
  //*/

  /*
  set<int> t;
  string err = "";
  for(int i=1; i<=n; i++){
    int key = 1+( rand()%(n*10) );
    auto node = t.find(key);
    if(node != t.end()) t.erase(node);
    else t.insert(key);
  }
  //*/

  return 0;
}