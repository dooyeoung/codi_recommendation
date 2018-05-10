# Fashion-Coordination Recommendation

**Project Site** [http://dooyeoung.ml](http://dooyeoung.ml/)
![webpage text](image/1.site.png)

---

> This is a web-site that recommend the fashion-coordination using MLP and QDA.
>
> - Recommand color-combination using MLP and Colormind API.
> - Evaluate the quality of color-combination using QDA.
> - Recommend the fashion by similarity of the input color-combination.


## 1. Code

##### skill

> - numpy, pandas, sklearn, keras, openCV
> - web : flask, sqlArchemy, bootstrap, MySQL

##### Code

>codi_flask : flask
>
>data : csv, sql file
>
>model : model(QDA, MLP) 
>
>utils : function
>
>- imageprocessing.py : Extract color
>- qda_goodbad.py : Evaluate the quality
>- recommand_color.py : Recommand colors
>
>Important jupyter notebook
>
>- EDA : Cluster, analysis_color
>- QDA : NaiveBayes
>- MLP : Recommand_color
>- Crawling : crawling_mapssi, download_image  



## 2. Method

### 2-1. Color Extraction

![alt text](image/2.extract.png)



### 2-2. EDA

- **Scatter**

![alt text](image/3.colorscatter.png)



- **Clustering (DBSCAN)**

![alt text](image/4.cluster.png)



### 2-3. Quality Evaluation (QDA)
- Generate mannered datas in random
- QDA Modeling for predict normal/mannered

![alt text](image/5.qda.png)



### 2-4. Color Recommendation

##### 2-4-1. **Overall method**

![alt text](image/6.recc.png)

- Recommended color 1 : The nearest color with predict color using colormind API
- Recommended color 2 : predict color
- Recommended color 3 : The second color with predict color using colormind API




##### 2-4-2. **MLP Model**

![alt text](image/7.mlp.png)



##### 2-4-3.**Colormind API**

The most predict-color is gray, so we use colormind API for variety of colors.

![alt text](image/8.capi.png)



### 2-5. Fashion Recommendation (KNN)

![alt text](image/9.knn.png)