import streamlit as st
import pandas as pd
# import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
pd.options.display.float_format = '{:.2f}'.format

## data
link_source = 'https://drive.google.com/uc?export=download&id='

link_product = link_source + '1QGEVPuV34xIfZMadexbnu3u1o4L_heYz'
link_review = link_source + '1Qd2j-SP0IZN_MOJ4lDbhN7dthDWKxdTS'
link_data_xl = link_source + '1Q5xnXFPHDENDfhjLx6RH1AYCkcz1y90b'
link_Recomender_Collborative = link_source + '1QSuaLQ8OInj3LAHl3aHMvNqrLwjZYHNL'



#--------------
# Gonfig GUI
st.set_page_config(page_title='Product Recommendation', layout = 'wide', initial_sidebar_state = "expanded",page_icon="📝")

# Load data
@st.cache
def load_products():
    return pd.read_csv(link_product)

products = load_products() 

@st.cache
def load_reviews():
    return pd.read_csv(link_review)

@st.cache
def load_data_xl():
    return pd.read_csv(link_data_xl, encoding='UTF-8')

@st.cache
def load_cosine_similarities():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
    tf = TfidfVectorizer(analyzer='word', min_df=0)
    tfidf_matrix = tf.fit_transform(data_xl.products_wt)
    return cosine_similarity(tfidf_matrix, tfidf_matrix)

@st.cache
def load_recomenders_collborative():
    return pd.read_csv(link_Recomender_Collborative)

# Function recommenders for an user
@st.cache
def get_recommenders_for_user(recommenders,customer_id, items_number=6):
    recds = recommenders[recommenders['customer_id']==customer_id]
    recds = recds.head(items_number)
    result = pd.merge(left=recds,right=products[products.item_id.isin(recds.product_id)] ,how='left',left_on='product_id',right_on='item_id')
    return result

# Set option show chart
st.set_option('deprecation.showPyplotGlobalUse', False)

#--------------
# GUI
#menu = ["Summary", "Model/Evaluate predict avocado prices", 'Predict avocado prices', "Time series"]
    #choice = st.sidebar.selectbox('Menu',menu)
choice = st.sidebar.radio("Menu",('Đế xuất dựa trên nội dung', 'Đề xuất dựa trên sản phẩm'))

if choice == "Đế xuất dựa trên nội dung":
    
    # Header-----
    #st.image('picture/hay.png')
    st.markdown("<h1 style='text-align: center; color: Yellow;'>Đế xuất dựa trên nội dung</h1>", unsafe_allow_html=True)
    st.write(" ")
    data_xl = load_data_xl()
    cosine_similarities = load_cosine_similarities()
    
    st.sidebar.markdown("<h3 style='text-align: center; color: Yellow;'>Chọn thông tin cần thiết</h3>", unsafe_allow_html=True)
    name_item = st.sidebar.selectbox('Tên sản phẩm', products['name'])
    #items_num = st.sidebar.slider(label='Số lượng sản phẩm đề xuất:',min_value=2,max_value=8,value=8,step=2)
    lst_type = [6,7,8,9,10,11,12]
    items_num = st.sidebar.selectbox('Số sản phẩm đề xuất', (lst_type))
    submit_button = st.sidebar.button(label='Đề xuất sản phẩm 📝')
    if submit_button:
        results = {}
        for idx, row in data_xl.iterrows():
            similar_indices = cosine_similarities[idx].argsort()[-9:-1]
            similar_items = [(cosine_similarities[idx][i]) for i in similar_indices]
            similar_items = [(cosine_similarities[idx][i], data_xl.index[i]) for i in similar_indices]
            results[idx] = similar_items[0:]
        ## lựa chọn sản phẩm
        st.markdown("<h3 style='text-align: left; color: Aqua;'>Thông tin sản phẩm</h3>", unsafe_allow_html=True)
        idx_prd = products.index[products['name']==name_item].tolist()[0]
        ## đưa hình vào form
        col1, col2 = st.columns([2,8])
        with col1:
            st.image(str(products.loc[idx_prd,'image']))
        with col2:
            st.write('###### %s'%(products.loc[idx_prd,'name']))
            strprice = '%s đ'%(format(products.loc[idx_prd,'price'],',d'))
            st.markdown("<h4 style='text-align: left; color: yellow;'>"+strprice+"</h4>", unsafe_allow_html=True)
            st.write('Brand: %s'%(products.loc[idx_prd,'brand']))
            st.write('Rating: %s'%(products.loc[idx_prd,'rating']))
            st.markdown("[Website](https://tiki.vn/"+products.loc[idx_prd,'url'].split('//')[-1]+")")
        st.write("")

        # Kết quả sản phẩm tương tự
        st.markdown("<h3 style='text-align: left; color: Aqua;'>Có thể bạn thích sản phẩm này không ?</h3>", unsafe_allow_html=True)
        sim_list = []
        for i in range(0,items_num,2):
            # for i
            item_id = data_xl.iloc[results[idx_prd][i][1]]['item_id']
            idx_prd = products.index[products['item_id']==item_id].tolist()[0]
            # for i+1
            item_id_1 = data_xl.iloc[results[idx_prd][i+1][1]]['item_id']
            idx_prd_1 = products.index[products['item_id']==item_id_1].tolist()[0]
            # in sản phẩm          
            col1, col2, col3, col4 = st.columns([2,3,2,3])
            with col1:
                st.image(str(products.loc[idx_prd,'image']))
            with col2:
                st.write('###### %s'%(products.loc[idx_prd,'name']))
                strprice = '%s đ'%(format(products.loc[idx_prd,'price'],',d'))
                st.markdown("<h4 style='text-align: left; color: yellow;'>"+strprice+"</h4>", unsafe_allow_html=True)
                st.write('Brand: %s'%(products.loc[idx_prd,'brand']))
                st.write('Rating: %s'%(products.loc[idx_prd,'rating']))
                st.markdown("[Website](https://tiki.vn/"+products.loc[idx_prd,'url'].split('//')[-1]+")")
            with col3:
                st.image(str(products.loc[idx_prd_1,'image']))
            with col4:
                st.write('###### %s'%(products.loc[idx_prd_1,'name']))
                strprice = '%s đ'%(format(products.loc[idx_prd_1,'price'],',d'))
                st.markdown("<h4 style='text-align: left; color: yellow;'>"+strprice+"</h4>", unsafe_allow_html=True)
                st.write('Brand: %s'%(products.loc[idx_prd_1,'brand']))
                st.write('Rating: %s'%(products.loc[idx_prd_1,'rating']))
                st.markdown("[Website](https://tiki.vn/"+products.loc[idx_prd_1,'url'].split('//')[-1]+")")
            st.write(" ")
    

elif choice == "Đề xuất dựa trên sản phẩm":
    #st.image('picture/Collaborative-filtering.jpg')
    st.markdown("<h1 style='text-align: center; color: Yellow;'>Đề xuất dựa trên sản phẩm</h1>", unsafe_allow_html=True)
    custIdsDefault = [18535485,11174386,17539237,16319435,11345636,15868853,13175665,10600682,13890346]
    recommenders = load_recomenders_collborative()
    with st.form(key='Đề xuất sản phẩm cho người dùng'):
        selected_user = st.sidebar.multiselect('Chọn người dùng', custIdsDefault ,[10600682])
        lst_type = [6,7,8,9,10,11,12]
        items_num = st.sidebar.selectbox('Số sản phẩm bạn muốn đế xuất:', (lst_type))
        #items_num = st.sidebar.slider(label='Số lượng sản phẩm đề xuất:',min_value=1,max_value=10,value=6,step=1)
        submit_button = st.sidebar.button(label='Chọn tôi đi 📝')
    if submit_button:
        if len(selected_user) ==0:
            st.sidebar.markdown("<h1 style='text-align: center; color: yellow;'>Chọn userid đi nào ?</h1>", unsafe_allow_html=True)
        else:
            data = get_recommenders_for_user(recommenders,selected_user[0],items_num)
            data.reset_index()
            #strtemp = 'Có {} Sản phẩm nào bán muốn nào:'.format(data.shape[0])
            st.markdown("<h3 style='text-align: left; color: Aqua;'>Sản phẩm để xuất cho bạn</h3>", unsafe_allow_html=True)
            # in sản phẩm   
            for i in range(0,data.shape[0],2):
                col1, col2, col3, col4 = st.columns([2,3,2,3])
                with col1:
                    st.image(str(data.loc[i,'image']))
                with col2:
                    st.write('###### %s'%(data.loc[i,'name']))
                    strprice = '%s đ'%(format(data.loc[i,'price'],',d'))
                    st.markdown("<h4 style='text-align: left; color: yellow;'>"+strprice+"</h4>", unsafe_allow_html=True)
                    st.write('brand: %s'%(data.loc[i,'brand']))
                    st.write('rating: %s'%(data.loc[i,'rating']))
                    st.markdown("[Website](https://tiki.vn/"+data.loc[i,'url'].split('//')[-1]+")")
                with col3:
                    if (i+1) < data.shape[0]:
                        st.image(str(data.loc[i+1,'image']))
                with col4:
                    if (i+1) < data.shape[0]:
                        st.write('###### %s'%(data.loc[i+1,'name']))
                        strprice = '%s đ'%(format(data.loc[i+1,'price'],',d'))
                        st.markdown("<h4 style='text-align: left; color: yellow;'>"+strprice+"</h4>", unsafe_allow_html=True)
                        st.write('brand: %s'%(data.loc[i+1,'brand']))
                        st.write('rating: %s'%(data.loc[i+1,'rating']))
                        st.markdown("[Website](https://tiki.vn/"+data.loc[i+1,'url'].split('//')[-1]+")")
                st.write("")