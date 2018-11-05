import pandas as pd
import nltk
import re
import config
from collections import Counter
from collections import defaultdict
import numpy as np


rake_stopword = pd.read_csv(config.path_dir.get('rake_stopwords'))
kwd = pd.read_csv(config.path_dir.get('keyword_data'))


tds_section_dict = {
	'Prof Fees/Individual':'194J',
	'Contractor - Others/Company':'194C',
	'Rent on Equipments/Individual':'194I',
	'Rent on Equipments/Company':'194I',
	'Contractor - Others/Individual':'194C',
	'Prof Fees/Company':'194J',
	'Contractor - Others/Firm':'194C',
	'Prof Fees/Firm':'194J',
	'Rent/Individual':'194I',
	'Rent/Company':'194I',
	'Rent/Firm':'194I',
	'Contractor - Others/HUF':'194C',
	'Rent/HUF':'194I',
	'Contractor - Others/Others':'194C',
	'Rent on Equipments/HUF':'194I',
	'Rent on Equipments/Firm':'194I',
	'Prof Fees/Others':'194J'
}

# making dict of stopwords.
stopwords = list(rake_stopword['words'])
stopwords = dict(Counter(stopwords))

to_extend_words = {'&':'and','i.e':'that is','ie':'that is',
					'e.g':'example','eg':'example'}



# sample cleaner function, this is a globar function which will clean the
# input string.

def cleaner(s):
	s = str(s)
	s = re.sub(r'\n',' ',s)
	for i in to_extend_words:
		if(re.search(r'\s%s\s'%(i), s)!=None):
			s = re.sub(r'%s'%(i),to_extend_words.get(i),s)
		else:
			continue
	s = re.sub(r'\(|\)|\d|[.,:;\'-]','',s)
	s = re.sub(r'\,',' ',s)
	s = re.sub(r'\s+',' ',s)
	s = s.lower()
	s = s.strip()
	return s

# making a dictionary of the keywords of the keywords.
kwd_info = defaultdict(list)
for i in range(0,kwd.shape[0]):
	s = str(kwd.loc[i,'Similar to description of invoice'])
	s = cleaner(s)
	kwd_info[s].append(int(kwd.loc[i,'HSN code']))
	kwd_info[s].append(str(kwd.loc[i,'Section number']))


class tds_scorer():
	def __init__(self):
		return
	def sub_stopwords(self,s):
		for i in stopwords:
			if(re.search(r'\s%s\s'%(i),s)!=None):
				s = re.sub(r'\s%s\s'%(i),' (stopword) ', s)
			if(re.search(r'^%s\s'%(i),s)!=None):
				s = re.sub(r'%s\s'%(i),'(stopword) ',s)
			if(re.search(r'\s%s$'%(i),s)!=None):
				s = re.sub(r'%s'%(i),' (stopword)',s)
			else:
				continue
		return s
	# splitting at stopword and making a list output.
	def rake_keywords(self,s):
		s = self.sub_stopwords(s)
		s = s.split('(stopword)')
		s = [i.strip() for i in s if(len(i)>1)]
		s = [i for i in set(s)]
		s = [i for i in s if(i!='')]
		return s
	# calculating frequency of each word in the rake_keyword list.
	def freq_words(self,s):
		s = self.rake_keywords(s)
		complete_text = ' '.join(s)
		return dict(Counter(complete_text.split(' ')))
	# calculating the degree for each keyword.
	def calculate_degree(self,s):
		word_degree = {}
		words = self.freq_words(s)
		keywords = self.rake_keywords(s)
		for i in words:
			count = 0
			for j in range(0,len(keywords)):
				if(i in keywords[j]):
					count+=1
				else:
					continue
			word_degree[i] = count
		return word_degree
	# calculating the score of each keywords by sum of degree(word)/freq(word).
	def score_word(self,s):
		keywords = self.rake_keywords(s)
		keyword_output_dict = {}
		freq = self.freq_words(s)
		degree = self.calculate_degree(s)
		for i in range(0,len(keywords)):
			score = 0
			word = keywords[i].split(' ')
			for j in range(0,len(word)):
				score = score + degree.get(word[j])/float(freq.get(word[j]))
			keyword_output_dict[keywords[i]] = score
		return keyword_output_dict
	# returning the word with the maximum score, array because in case we have two same score keywords .
	def max_score_word(self,s):
		keywords = self.score_word(s)
		all_scores = [i for i in keywords.values()]
		if(all_scores!=[]):
			max_score = max(all_scores)
			relev_keywords = [key for key, value in keywords.items() if(value==max_score)]
			relev_keywords = [i for i in set(relev_keywords)]
		else:
			relev_keywords = []
		return relev_keywords
	# for handling nagattion.
	# this method will only be called for the case when we will have failure in 
	# desciption field.
	def processing_naggation(self, index, dataframe_path_input):
		train = pd.read_csv(dataframe_path_input, usecols=['Supplier PAN No','TDS Type','HSN codes under GST','AP Narration'])
		temp_ind = index
		s = str(train.loc[temp_ind,'AP Narration'])
		s = cleaner(s)
		if(((str(train.loc[temp_ind,'HSN codes under GST']))!='nan')and(str(train.loc[temp_ind,'TDS Type'])in tds_section_dict.keys())):
			section_code = str(tds_section_dict.get(str(train.loc[temp_ind,'TDS Type'])))
			hsn_code = int(train.loc[temp_ind,'HSN codes under GST'])
			keywords_string = self.max_score_word(s)
			keywords_matched = []
			for j in range(0,len(keywords_string)):
				keyword = keywords_string[j].split(' ')
				for k in range(0,len(keyword)):
					for key, value in kwd_info.items():
						if((keyword[k] in key)&(value[0]==hsn_code)&(value[1]==section_code)):
							keywords_matched.append(cleaner(key))
							break
						else:
							continue
			keywords_matched = [ind for ind in set(keywords_matched)]
			for ind in range(0,len(keywords_matched)):
				keyword = str(keywords_matched[ind]).split(' ')
				score = 0.0
				for j in range(0,len(keyword)):
					if(keyword[j] in s):
						score = score + 1/(float(len(keyword)))
					else:
						continue
				keywords_matched[ind] = (keywords_matched[ind],) + (score,)
			max_kwd_score = [keywords_matched[i][1] for i in range(0,len(keywords_matched))]
			if(max_kwd_score!=[]):
				max_kwd_score = max(max_kwd_score)
				keywords_matched = [keywords_matched[i] for i in range(0,len(keywords_matched)) if(keywords_matched[i][1]==max_kwd_score)]
			else:
				keywords_matched = []
			keywords_matched = [keywords_matched[i][0] for i in range(0,len(keywords_matched))]
			if(len(keywords_matched)!=0):
				#print(keywords_matched[0])
				#print(kwd_info.get(keywords_matched[0]))
				kwd_find = keywords_matched[0]
				section = kwd_info.get(kwd_find)[1]
				hsn_number = kwd_info.get(kwd_find)[0]
				# Now, making the if else statements from the logic behing the found section
				if(section=='194C'):
					pan_number_supplier = str(train.loc[i,'Supplier PAN No'])
					if(pan_number_supplier[3] in ['H','P']):
						return "1"
					else:
						return "2"
				elif(section=='194J'):
					return "10"
				elif(section=='194I'):
					if((hsn_number==9963)or(hsn_number==9972)):
						return "10"
					elif((hsn_number==9973)or(hsn_number==9966)or(hsn_number==9985)):
						return "2"
				else:
					return False
			else:
				return False
		else:
			return False
	# handling the simple cases first and than i will go into nagattion.
	def processing(self,start,end, dataframe_path_input):
		train = pd.read_csv(dataframe_path_input)
		temp_ind = start
		for i in range(start, end):
			s = str(train.loc[temp_ind,'Description'])
			s = cleaner(s)
			# checking for not null fields of TDS Type and Hsn code field.
			if((str(train.loc[temp_ind,'TDS Type']) in tds_section_dict.keys())and(str(train.loc[temp_ind,'HSN codes under GST']))):
				section_code = str(tds_section_dict.get(str(train.loc[temp_ind,'TDS Type'])))
				hsn_code = int(train.loc[temp_ind,'HSN codes under GST'])
				keywords_string = self.max_score_word(s)
				keywords_matched = []
				for j in range(0,len(keywords_string)):
					keyword = keywords_string[j].split(' ')
					for k in range(0,len(keyword)):
						for key, value in kwd_info.items():
							if((keyword[k] in key)&(value[0]==hsn_code)&(value[1]==section_code)):
								keywords_matched.append(cleaner(key))
								break
							else:
								continue
				keywords_matched = [ind for ind in set(keywords_matched)]
				# Hence, this block will give me a high score keywords only so that i can use them 
				# only while getting the keywords for tds rate determination.
				for ind in range(0,len(keywords_matched)):
					keyword = str(keywords_matched[ind]).split(' ')
					score = 0.0
					for j in range(0,len(keyword)):
						if(keyword[j] in s):
							score = score + 1/(float(len(keyword)))
						else:
							continue
					keywords_matched[ind] = (keywords_matched[ind],) + (score,)
				max_kwd_score = [keywords_matched[i][1] for i in range(0,len(keywords_matched))]
				if(max_kwd_score!=[]):
					max_kwd_score = max(max_kwd_score)
					keywords_matched = [keywords_matched[i] for i in range(0,len(keywords_matched)) if(keywords_matched[i][1]==max_kwd_score)]
				else:
					keywords_matched = []
				keywords_matched = [keywords_matched[i][0] for i in range(0,len(keywords_matched))]
				# Hence, now i have got all the relevant keywords from my text and i wil now simply verify
				# it from  my sample data
				if(len(keywords_matched)!=0):
					#print(keywords_matched[0])
					#print(kwd_info.get(keywords_matched[0]))
					kwd_find = keywords_matched[0]
					section = kwd_info.get(kwd_find)[1]
					hsn_number = kwd_info.get(kwd_find)[0]
					# Now, making the if else statements from the logic behing the found section
					if(section=='194C'):
						pan_number_supplier = str(train.loc[i,'Supplier PAN No'])
						if(pan_number_supplier[3] in ['H','P']):
							train.loc[temp_ind,'tds_ml_found'] = "1"
						else:
							train.loc[temp_ind,'tds_ml_found'] = "2"
					elif(section=='194J'):
						train.loc[temp_ind,'tds_ml_found'] = "10"
					elif(section=='194I'):
						if((hsn_number==9963)or(hsn_number==9972)):
							train.loc[temp_ind,'tds_ml_found'] = "10"
						elif((hsn_number==9973)or(hsn_number==9966)or(hsn_number==9985)):
							train.loc[temp_ind,'tds_ml_found'] = "2"
					else:
						train.loc[temp_ind,'tds_ml_found'] = np.NaN
				else:
					elem = self.processing_naggation(temp_ind, dataframe_path_input)
					#elem = self.processing_naggation(str(train.loc[temp_ind,'AP Narration']), hsn_code, section_code)
					if(elem==False):
						train.loc[temp_ind,'tds_ml_found'] = np.NaN
					else:
						train.loc[temp_ind,'tds_ml_found'] = str(elem)
			# If that is not the case than return nan and continue
			else:
				train.loc[temp_ind,'tds_ml_found'] = np.NaN
			temp_ind+=1
		return pd.DataFrame(train.loc[range(start, end),['tds_ml_found']])






