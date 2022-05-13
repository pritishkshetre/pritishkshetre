### Hola, I'm Pritish! 👋


- 🔭 I’m currently working on virtual internship programs.
- 🌱 I’m currently learning new technologies.
- 💬 Ask me about business insights.
- 📫 How to reach me: 
-                      Twitter - @pritishkshetre
-                      Instagram - @pritishkshetre
-                      Facebook - @pritishkshetre
-                      Linked In - @pritish-kshetre
- 😄 Pronouns: He/His
- ⚡ Fun fact: I laugh a lot. 

Experiment 1

Experiment 4  install pseudo_facebook.csv and spotify.csv(genres_v2.csv)
import pandas as pd
data = {"Roll-num": [10,20,30,40,50,60,70], "Age":[12,10,14,16,18,19,15], 
        "Name":['Joy','Joseph','Joyce','Alexa','Siri','Jarvis','Jay']}
block = pd.DataFrame(data)
print("Original Data frame:\n")
print(block)
from google.colab import files
upload=files.upload()
import pandas as pd 
df=pd.read_csv("pseudo_facebook.csv")
print(df)
from google.colab import files
upload=files.upload()
import pandas as pd 
df=pd.read_csv("genres_v2.csv")
print(df)
block.loc[[0,1,3]]
block.iloc[[0,1,3,6],[0,2]]
import pandas as pd
df=pd.concat(map(pd.read_csv,["pseudo_facebook.csv","genres_v2.csv"]))
print(df)
import pandas as pd
df.sort_values(by="age",ascending=False)
import pandas as pd 
df_transposed=df.T
print(df)



Experiment 5   install datacl.csv
import pandas as pd
import numpy as np
from google.colab import files
upload=files.upload()
df=pd.read_csv("datacl.csv")
print(df.shape)
print(df.info())
//droppingduplicates//
df=df.drop_duplicates('x1',keep='first')
print(df.shape)
//missingvalues//
df=df.isnull().sum()
print(df)
//datatransformation//
df=pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), columns=['a','b','c'])
df.transform(func = lambda x : x * 10)
