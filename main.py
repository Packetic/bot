import pandas as pd
import numpy as np
import nltk

from aiogram import Bot, types
from aiogram.dispatcher import Dispatcher
from aiogram.utils import executor

from config import TOKEN

from sklearn.model_selection import train_test_split as train
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

# работа с нейросетью и анализом данных

df = pd.read_excel('/home/pc-08/Рабочий стол/DataSet2021Spring.xlsx')
new_df = df['Number_PP'].str.split(',', expand=True)

# новые имена колонок
new_df.columns = ['Number_PP', 'Name', 'Full_name', 'Code', 'Unit_of_reconciliation',
                  'Additional_characteristics', 'Owners_name', 'Kind', 'Article',
                  'Weight', 'Accounting_characteristics', 'Storage_unit', 'VAT_rate',
                  'Service', 'Nomenclature_group', 'Sharto', 'Main_supplier',
                  'Main_supplier_object', 'Comment', 'Predefined', 'Price', 'Qty_stock',
                  'Remaining_amount', 'Currency', 'Fe', 'To_change', 'Non', 'Non', 'Non', 'Non', 'Non', 'Non']

# выделение оптимального датасета
good_df = new_df.loc[new_df['To_change'] == 'Руб']

# немного ручного труда
good_df['Price'] = good_df['Price'] + ',' + good_df['Qty_stock']
good_df['Qty_stock'] = good_df['Remaining_amount']
good_df['Remaining_amount'] = good_df['Currency']
good_df['Currency'] = good_df['Fe']
good_df['Fe'] = good_df['To_change']
good_df['To_change'] = good_df['Non']

good_df['Qty_stock'] = good_df['Qty_stock'] + ',' + good_df['Remaining_amount']
good_df['Remaining_amount'] = good_df['Currency']
good_df['Currency'] = good_df['Fe']
good_df['Fe'] = good_df['To_change']

# переименовываем колонки, финальный вид
good_df.columns = ['Number_PP', 'Name', 'Full_name', 'Code', 'Unit_of_reconciliation',
                   'Additional_characteristics', 'Owners_name', 'Kind', 'Article',
                   'Weight', 'Accounting_characteristics', 'Storage_unit', 'VAT_rate',
                   'Service', 'Nomenclature_group', 'Sharto', 'Main_supplier',
                   'Main_supplier_object', 'Comment', 'Predefined', 'Price', 'Qty_stock',
                   'Remaining_amount', 'Currency', 'Non', 'Non', 'Non', 'Non', 'Non', 'Non', 'Non', 'Non']


le = LabelEncoder()
ohe = OneHotEncoder(handle_unknown='ignore')

# избавляемся от лишней информации
for i in good_df.columns[:24]:
    good_df[i] = ohe.fit_transform(good_df[i])
X = good_df.values[::, 1:23].astype('int64')
y = good_df.values[::, 0:1].astype('int64')
test = SelectKBest(score_func=chi2, k=4)
fit = test.fit(X, y)
np.set_printoptions(precision=3)
# print(fit.scores_)
features = fit.transform(X)
print(features[0:5, :])
print(features)

# начинаем само обучение
X_train, X_test, y_train, y_test = train(features, y, test_size=0.75)
clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
clf.fit(X_train, y_train)
clf.score(X_test, y_test)


# сам бот

bot = Bot(token=TOKEN)
dp = Dispatcher(bot)


@dp.message_handler(commands=['start'])
async def process_start_command(message: types.Message):
    await message.reply("Здравствуйте, напишите, что вы бы хотели посмотреть в каталоге товаров")


@dp.message_handler(commands=['help'])
async def process_help_command(message: types.Message):
    await message.reply("")


@dp.message_handler()
async def echo_message(msg: types.Message):
    await bot.send_message(msg.from_user.id, msg.text)


if __name__ == '__main__':
    executor.start_polling(dp)
