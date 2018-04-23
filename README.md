# Fashion-Coordination Recommendation

> This is a web-site that recommend the fashion-coordination using MLP and QDA.
>
> - Recommand color-combination using MLP and Colormind API.
> - Evaluate the quality of color-combination using QDA.
> - Recommend the fashion by similarity of the input color-combination.

**Web-site address** [link](http://www.mdpi.com/2073-8994/10/1/4) 



![webpage text](https://github.com/wjy5446/codi_recommendation/blob/master/image/webpage.png)



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

![alt text](https://github.com/wjy5446/codi_recommendation/blob/master/image/extract_color.png)





### 2-2. EDA

- **Scatter**

![alt text](https://github.com/wjy5446/codi_recommendation/blob/master/image/EDA.png)



- **Clustering (DBSCAN)**

![alt text](https://github.com/wjy5446/codi_recommendation/blob/master/image/clustering.png)



### 2-3. Quality Evaluation (QDA)

![alt text](https://github.com/wjy5446/codi_recommendation/blob/master/image/Quality_Evaluation.png)



### 2-4. Color Recommendation

- **Overall method**

![alt text](https://github.com/wjy5446/codi_recommendation/blob/master/image/Recommend_color.png)

- Recommended color 1 : The nearest color with predict color using colormind API
- Recommended color 2 : predict color
- Recommended color 3 : The second color with predict color using colormind API





- **MLP Model**

![alt text](https://github.com/wjy5446/codi_recommendation/blob/master/image/MLP_model.png)






- **Colormind API**

The most predict-color is gray, so we use colormind API for variety of colors.

![alt text](https://github.com/wjy5446/codi_recommendation/blob/master/image/Colormind_API.png)



### 2-5. Fashion Recommendation (KNN)

![alt text](https://github.com/wjy5446/codi_recommendation/blob/master/image/Recommend_fashion.png)