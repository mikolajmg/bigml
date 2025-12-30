import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer

def gini_index(y):
    if len(y) == 0:
        return 0
    
    counts = np.bincount(y)
    probs = counts[counts > 0] / len(y)
    return 1.0 - np.sum(probs**2)


class TreeNode:
    def __init__(self,depth,vocab_length,is_root,value=None,left=None,right=None):
        self.depth = depth # na jakim poziomie drzewa sie znajduje
        self.left = left # lewy potomek
        self.right = right # prawy potomek
        self.value = value # wartosc klasyfikacji jezeli jest lisciem
        self.vocab_length = vocab_length # caly zbior slow
        self.vocab_used = random.sample(list(range(self.vocab_length)),k=int(self.vocab_length**0.5)) # podzbior slow do rozpatrzenia na tym wezle
        self.is_root = is_root # czy ma potomków
        self.eps = 1e-9
        
        
    def get_split_mask(self, X, word_id):
        
        return (X[:, word_id].toarray().flatten() != 0)
    
    def split(self,X=None,y=None,word_id=None):
        X_left, y_left = X[(X[:,word_id]!=0).toarray().flatten()], y[(X[:,word_id]!=0).toarray().flatten()]
        X_right, y_right =X[(X[:,word_id]!=1).toarray().flatten()], y[(X[:,word_id]!=1).toarray().flatten()]
    
        return  X_left, y_left,X_right, y_right
    

    def annotate(self,X=None,y=None):
        # TODO Its possible to not iterate twice over gini calculation in left and right subsets 
        if isinstance(self.value,int) or isinstance(self.value,float):
            Exception("Leaf node was aleady assigned a value. Do you really wanted to annotare it again?")
        if self.depth ==10:
            self.value = y.mode()[0]
            self.is_root = True
            self.word = None
            return self.value
            
            
        Gini = gini_index(y)
        if Gini < self.eps:
            self.value = y.mode()[0]
            self.is_root = True
            self.word = None
            return self.value
        
        max_gain = -1
        for word in tqdm(self.vocab_used, desc=f"Searching best split at node depth {self.depth}",leave=False):
            mask = self.get_split_mask(X, word_id=word)
            #X_left, y_left,X_right, y_right = self.split(X,y,word_id=word)
            y_left = y[mask]
            y_right = y[~mask]

            gini_left = gini_index(y_left)
            gini_left*= (len(y_left)/len(y))
            
            gini_right = gini_index(y_right)
            gini_right*= (len(y_right)/len(y))
            
            gain = Gini - (gini_left + gini_right)
            

            if gain>max_gain:
                max_gain = gain
                self.word = word
                best_mask = mask
                #print(f"Depth: {self.depth}, Chosen word: {self.word}, Gain: {max_gain}")
        if max_gain < self.eps:
            self.value = y.mode()[0]
            self.is_root = True
            self.word = None
            return self.value
            
        X_left, y_left,X_right, y_right = X[best_mask], y[best_mask],X[~best_mask], y[~best_mask]
        self.left = TreeNode(self.depth+1,self.vocab_length,False)
        self.right = TreeNode(self.depth+1,self.vocab_length,False)


        if self.word is None:
            return self.value
        else:
            return [self.word,self.left.annotate(X_left, y_left),self.right.annotate(X_right, y_right)]
        
        
        
class TREE_CLASSIFIER:
    def __init__(self,list_of_words,max_depth=10,output_file="test.txt",seed=42):
        self.V = list_of_words
        
        self.root = TreeNode(0,self.V,False)
        self.max_depth = max_depth
        self.class_list = None
        self.output_file = output_file

    def annotate(self,X,y):
        self.class_list = self.root.annotate(X,y)
        with open(self.output_file, "a") as myfile:
            myfile.write(str(self.class_list)+"\n")
        


class MAIN_WORKER:
    def __init__(self,dataset_path,model_output,n_trees=10,seed=42):
        self.dataset_path= dataset_path
        
        self.model_output = model_output
        self.n_trees = n_trees

    def create_vocabulary(self):
        data =  pd.read_csv(self.dataset_path,header=None,nrows=1000)
        data.columns = ["review","rating"]
        data.replace(to_replace=r'[^a-zA-Z\s]', value='', regex=True, inplace=True)
        data['review'] = data['review'].str.lower()
        self.vectorizer = CountVectorizer(min_df=2)
        self.X = self.vectorizer.fit_transform(data["review"])
        self.y = data["rating"]

        with open(self.model_output, "a") as myfile:
            myfile.write(' '.join(self.vectorizer.get_feature_names_out())+"\n")

    def train_forest(self):
        list_of_words = len(list(self.vectorizer.get_feature_names_out()))
        for i in range(self.n_trees):
            tree = TREE_CLASSIFIER(list_of_words,output_file=self.model_output,seed=i)
            tree.annotate(self.X,self.y)

if __name__ == "__main__":
    worker = MAIN_WORKER("amazon_reviews_2M.csv","model.txt",n_trees=10)
    worker.create_vocabulary()
    worker.train_forest()