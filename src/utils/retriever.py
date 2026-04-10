import os
from languagebind import LanguageBind, to_device, transform_dict, LanguageBindImageTokenizer, LanguageBindVideoTokenizer
import torch
import numpy as np
import cv2
import pickle
import time
import json
from moviepy import VideoFileClip
from decord import VideoReader, cpu
from tqdm import tqdm

import math
import argparse
from video_utils import *
import subprocess
import datetime
import multiprocessing

from FlagEmbedding import BGEM3FlagModel



class Retrieval_Manager():
    def __init__(self, args=None, batch_size=1, clip_save_folder=None, clip_duration=10,dataset_folder='dataset/CG-Bench',retrievl_device='cuda:0'):
        self.device='cuda'
        # if args.retriever_type=='large':
        path = 'models/LanguageBind_Video_FT'
        # elif args.retriever_type=='huge':
        #     path = 'LanguageBind/LanguageBind_Video_Huge_V1.5_FT'
        # else:
        #     raise KeyError
        clip_type = {
            'video':  'models/LanguageBind_Video_FT', # also LanguageBind_Video
            'image': 'models/LanguageBind_Image'
        }
        self.model = LanguageBind(clip_type=clip_type,device='cuda').to('cuda')
        self.text_retriever = BGEM3FlagModel('models/bge-m3', use_fp16=True,devices=['cuda']) # Setting use_fp16 to True speeds up computation with a slight performance degradation
        self.model.eval()
        self.tokenizer = LanguageBindVideoTokenizer.from_pretrained(path)
        self.modality_transform = {c: transform_dict[c](self.model.modality_config[c]) for c in clip_type.keys()}
        self.clip_embs_cache = {}
        self.frame_embs_cache = {}
        self.batch_size = 1
        self.clip_save_folder = clip_save_folder
        self.args=args
        self.clip_duration = 10
        self.dataset_folder=dataset_folder
        self.retriever_type='large'
    def load_model_to_device(self, device):

        self.model.to(device)

        def recursive_to(module):
            for name, attr in module.__dict__.items():
                if isinstance(attr, torch.nn.Module):
                    attr.to(device)
                    recursive_to(attr)
                elif isinstance(attr, torch.Tensor):
                    setattr(module, name, attr.to(device))
                elif isinstance(attr, (list, tuple)):
                    new_attrs = []
                    for item in attr:
                        if isinstance(item, torch.nn.Module):
                            item.to(device)
                            recursive_to(item)
                        elif isinstance(item, torch.Tensor):
                            item = item.to(device)
                        new_attrs.append(item)
                    setattr(module, name, type(attr)(new_attrs))

        recursive_to(self.model)

    def load_model_to_cpu(self):
        self.device=torch.device('cpu')
        self.load_model_to_device(torch.device('cpu'))
    
    def load_model_to_gpu(self, gpu_id=0):
        # self.device = torch.device(gpu_id)
        self.model.to('cuda')

    def cut_video(self, video_path, clip_save_folder=None, total_duration=-1):
        valid_clip_paths = set()
        time1 = time.time()
        os.makedirs(clip_save_folder, exist_ok=True)

        duration = VideoFileClip(video_path).duration
        chunk_number = math.ceil(duration/self.clip_duration)

        total_video_clip_paths = []
        for i in range(chunk_number):
            start_time = self.clip_duration * i
            end_time = start_time + self.clip_duration
            output_filename = f'clip_{i}_{self.format_time(start_time)}_to_{self.format_time(end_time)}.mp4'  
            total_video_clip_paths.append(clip_save_folder+'/'+output_filename)     

        if os.path.exists(clip_save_folder):
            valid_clip_num = 0
            path_li = os.listdir(clip_save_folder)
            for clip_name in path_li:
                try:
                    VideoReader(clip_save_folder+'/'+clip_name, ctx=cpu(0), num_threads=1)
                    valid_clip_paths.add(clip_save_folder+'/'+clip_name)
                    valid_clip_num+=1
                    del total_video_clip_paths[total_video_clip_paths.index(clip_save_folder+'/'+clip_name)]
                except Exception as e: 
                    os.system('rm -rf '+clip_save_folder+'/'+clip_name) # 移除不合法的clip
                    
            # assert valid_clip_num >= chunk_number-5,f'valid_clip_num:{valid_clip_num} < chunk_number-5: {chunk_number-3}, clip_save_folder:{clip_save_folder}'
            return [file for file in sorted(valid_clip_paths, key=lambda x: int(x.split('/')[-1].split('_')[1]))]
        
        else:
            print(clip_save_folder,'no valid clips found, cutting video:', video_path)
        
        return sorted(list(valid_clip_paths), key=lambda x: int(x.split('/')[-1].split('_')[1]))

    def save_clip(self, clip, clip_save_folder, clip_index, start_time, end_time, fps):
        start_time_str = self.format_time(start_time)
        end_time_str = self.format_time(end_time)
        os.makedirs(clip_save_folder,exist_ok=True)
        clip_path = os.path.join(clip_save_folder, f"clip_{clip_index}_{start_time_str}_to_{end_time_str}.mp4")
        height, width, _ = clip[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(clip_path, fourcc, fps, (width, height))

        for frame in clip:
            out.write(frame)

        out.release()
        return clip_path

    def format_time(self, seconds):
        mins, secs = divmod(seconds, 60)
        hours, mins = divmod(mins, 60)
        return f"{int(hours):02d}-{int(mins):02d}-{int(secs):02d}"

    def parse_time(self, time_str):
        hours, mins, secs = map(int, time_str.split('-'))
        total_seconds = hours * 3600 + mins * 60 + secs
        return total_seconds


    @ torch.no_grad()
    def calculate_video_clip_embedding(self, video_path, folder_path, total_duration=None):
        total_embeddings = []
        video_name = video_path.split('/')[-1].split('.')[0]

        folder_path = f'{self.dataset_folder}/embeddings/{self.clip_duration}/{self.retriever_type}/'
        os.makedirs(folder_path,exist_ok=True)

        embedding_path = os.path.join(folder_path,video_name+'.pkl')
        clip_path = os.path.join(folder_path,video_name+'_clip_paths.pkl')
       
        if os.path.exists(embedding_path) and os.path.exists(clip_path):
            video_paths = pickle.load(open(clip_path,'rb'))
            total_embeddings = pickle.load(open(embedding_path,'rb')).cpu()

            if len(video_paths) > total_duration //self.clip_duration - 3:
                print('existing the emebdding')
                return video_paths, total_embeddings
            else:
                print(video_paths,total_duration, self.clip_duration,len(embedding_path),embedding_path,'exist but have not enough valid video number!!')
        # print('calculating video embeddings',self.clip_save_folder,self.clip_duration,1234)
        video_paths = self.cut_video(video_path, os.path.join(self.clip_save_folder,video_path.split('/')[-1].split('.')[0]),total_duration)
        p = os.path.join(self.clip_save_folder,video_path.split('/')[-1].split('.')[0])
        # print(video_paths)
        try:
            video_paths=split_video_to_clips(video_path,p,clip_duration=10)[0]
            print('split videos')
        except:
             print('error,ffmpeg',video_path)
        #     # print(embedding_path,clip_path,1234)
             video_paths=[video_path]
        # print(a)
        # print('split video',video_path)
        assert len(video_paths) != 0, f'folder {p} have no valid clips'
        total_embeddings = []
        valid_video_paths = []
        for i in range(len(video_paths)):
            try:
                inputs = {'video': to_device(self.modality_transform['video'](video_paths[i]), self.device)}
                with torch.no_grad():
                    embeddings = self.model(inputs)
                    valid_video_paths.append(video_paths[i])
                    total_embeddings.append(embeddings['video'])
            except Exception as e:
                print(e)
            torch.cuda.empty_cache()
        total_embeddings = torch.cat(total_embeddings,dim=0)
        os.makedirs(folder_path,exist_ok=True)
        pickle.dump(total_embeddings,open(f'{folder_path}/{video_name}.pkl','wb'))
        pickle.dump(valid_video_paths,open(f'{folder_path}/{video_name}_clip_paths.pkl','wb'))
        # print(1234567,video_paths,total_embeddings)
        return video_paths,total_embeddings






    @ torch.no_grad()
    def calculate_text_embedding(self,text,video_path=None,flag_save_embedding=True):
        if flag_save_embedding:
            video_name = video_path.split('/')[-1].split('.')[0]
            os.makedirs(f'{self.dataset_folder}/embeddings/subtitle/{self.retriever_type}',exist_ok=True)
            embedding_path = f'{self.dataset_folder}/embeddings/subtitle/{self.retriever_type}/{video_name}_subtitle.pkl'
            try:
                embeddings = pickle.load(open(embedding_path,'rb'))
                # print('use precalculated subtitle embeddings')
                return embeddings
            except:
                pass

        # print('calculating subtitle embeddings')
        inputs = {'language':to_device(self.tokenizer(text, max_length=77, padding='max_length',truncation=True, return_tensors='pt'), self.device)}

        with torch.no_grad():
            embeddings = self.model(inputs)
        if flag_save_embedding:
            pickle.dump(embeddings['language'],open(embedding_path,'wb'))
        torch.cuda.empty_cache()
        return embeddings['language']



    def get_informative_clips(self,query,video_path,top_k=0,similarity_threshold=-100,topk_similarity=0,total_duration=-1,return_score=False):
        torch.cuda.empty_cache()
        if ".mp4" not in video_path:
            video_path, query = query, video_path

        # top_k=5
        assert top_k!=0 and similarity_threshold==-100 and topk_similarity==0 or top_k==0 and similarity_threshold!=-100 and topk_similarity==0 or top_k==0 and similarity_threshold==-100 and topk_similarity!=0,f'only one of top_k and simlarity_threshold should be assigned!'

        if similarity_threshold!=-100 or topk_similarity!=0:
            top_k=100

        # Calculate and normalize the query embedding
        q_emb = self.calculate_text_embedding(query,flag_save_embedding=False).cpu()
        q_emb = q_emb / q_emb.norm(p=2, dim=1, keepdim=True)

        if video_path not in self.clip_embs_cache:
            if len(self.clip_embs_cache) > 1:  # Only keep cache for one video
                self.clip_embs_cache = {}
            video_name = video_path.split('/')[-1].split('.')[0]
            folder_path = f'{self.dataset_folder}/embeddings/{self.clip_duration}/{self.retriever_type}'
            video_clip_paths, clip_embs = self.calculate_video_clip_embedding(video_path, folder_path, total_duration)
            if type(clip_embs)==dict:
                clip_embs = clip_embs['video']

            clip_embs = clip_embs.cpu()
            self.clip_embs_cache[video_path] = video_clip_paths, clip_embs
        else:
            video_clip_paths, clip_embs = self.clip_embs_cache[video_path]
        clip_embs = clip_embs / clip_embs.norm(p=2, dim=1, keepdim=True)
        similarities = torch.matmul(q_emb, clip_embs.T)

        # Get the indices of the top_k clips
        top_k_indices = similarities[0].argsort(descending=True)[:top_k].tolist()
        # print(123,top_k_indices,video_clip_paths)
        # Return list of tuples (path, similarity score) with similarity above threshold
        result = []
        
        for i in top_k_indices:
            sim_score = similarities[0][i].item()
            # print(sim_score)
            if sim_score > similarity_threshold:
                result.append((video_clip_paths[i], sim_score))
        
        torch.cuda.empty_cache()
        if top_k==0:
            result = result[:10] # 最多10个clip
        # return [(0,total_duration)],[video_path]
     
        return parse_and_sort_file_paths(result)
    





    @ torch.no_grad()
    def get_informative_clips_with_video_query(self,query, query_video_path,video_path,top_k=0,similarity_threshold=-100,topk_similarity=0,total_duration=-1,return_score=False):
        torch.cuda.empty_cache()
        assert top_k!=0 and similarity_threshold==-100 and topk_similarity==0 or top_k==0 and similarity_threshold!=-100 and topk_similarity==0 or top_k==0 and similarity_threshold==-100 and topk_similarity!=0,f'only one of top_k and simlarity_threshold should be assigned!'

        # if similarity_threshold!=-100 or topk_similarity!=0:
        #     top_k=100

        # Calculate and normalize the query embedding
        text_emb = self.calculate_text_embedding(query,flag_save_embedding=False).cpu()
        text_emb = text_emb / text_emb.norm(p=2, dim=1, keepdim=True)

        inputs = {'video': to_device(self.modality_transform['video'](query_video_path), self.device)}
        with torch.no_grad():
            q_emb = self.model(inputs)['video'].cpu()
        q_emb = q_emb / q_emb.norm(p=2, dim=1, keepdim=True)

        q_emb = q_emb + text_emb

        if video_path not in self.clip_embs_cache:
            if len(self.clip_embs_cache) > 1:  # Only keep cache for one video
                self.clip_embs_cache = {}
            video_name = video_path.split('/')[-1].split('.')[0]
            folder_path = f'{self.dataset_folder}/embeddings/{self.clip_duration}/{self.retriever_type}'
            video_clip_paths, clip_embs = self.calculate_video_clip_embedding(video_path, folder_path, total_duration)
            if type(clip_embs)==dict:
                clip_embs = clip_embs['video']

            clip_embs = clip_embs.cpu()
            self.clip_embs_cache[video_path] = video_clip_paths, clip_embs
        else:
            video_clip_paths, clip_embs = self.clip_embs_cache[video_path]

        # Normalize the clip embeddings
        clip_embs = clip_embs / clip_embs.norm(p=2, dim=1, keepdim=True)

        # Compute similarities
        similarities = torch.matmul(q_emb, clip_embs.T)

        # Get the indices of the top_k clips
        top_k_indices = similarities[0].argsort(descending=True)[:top_k].tolist()

        # Return list of tuples (path, similarity score) with similarity above threshold
        # result = []
        
        for i in top_k_indices:
            sim_score = similarities[0][i].item()
            # print(sim_score)
            if sim_score > similarity_threshold:
                result.append((video_clip_paths[i], sim_score))
        
        torch.cuda.empty_cache()
        if top_k==0:
            result = result[:10] # 最多10个clip
        return result



    @ torch.no_grad()
    def get_informative_clips(self,query,video_path,top_k=0,similarity_threshold=-100,topk_similarity=0,total_duration=-1,return_score=False):
        torch.cuda.empty_cache()
        if ".mp4" not in video_path:
            video_path, query = query, video_path

        # top_k=5
        assert top_k!=0 and similarity_threshold==-100 and topk_similarity==0 or top_k==0 and similarity_threshold!=-100 and topk_similarity==0 or top_k==0 and similarity_threshold==-100 and topk_similarity!=0,f'only one of top_k and simlarity_threshold should be assigned!'

        if similarity_threshold!=-100 or topk_similarity!=0:
            top_k=100

        # Calculate and normalize the query embedding
        q_emb = self.calculate_text_embedding(query,flag_save_embedding=False).cpu()
        q_emb = q_emb / q_emb.norm(p=2, dim=1, keepdim=True)

        if video_path not in self.clip_embs_cache:
            if len(self.clip_embs_cache) > 1:  # Only keep cache for one video
                self.clip_embs_cache = {}
            video_name = video_path.split('/')[-1].split('.')[0]
            folder_path = f'{self.dataset_folder}/embeddings/{self.clip_duration}/{self.retriever_type}'
            video_clip_paths, clip_embs = self.calculate_video_clip_embedding(video_path, folder_path, total_duration)
            if type(clip_embs)==dict:
                clip_embs = clip_embs['video']

            clip_embs = clip_embs.cpu()
            self.clip_embs_cache[video_path] = video_clip_paths, clip_embs
        else:
            video_clip_paths, clip_embs = self.clip_embs_cache[video_path]
        clip_embs = clip_embs / clip_embs.norm(p=2, dim=1, keepdim=True)
        similarities = torch.matmul(q_emb, clip_embs.T)

        # Get the indices of the top_k clips
        top_k_indices = similarities[0].argsort(descending=True)[:top_k].tolist()
        # print(123,top_k_indices,video_clip_paths)
        # Return list of tuples (path, similarity score) with similarity above threshold
        result = []
        
        for i in top_k_indices:
            sim_score = similarities[0][i].item()
            # print(sim_score)
            if sim_score > similarity_threshold:
                result.append((video_clip_paths[i], sim_score))
        
        torch.cuda.empty_cache()
        if top_k==0:
            result = result[:10] # 最多10个clip
        # return [(0,total_duration)],[video_path]
     
        return parse_and_sort_file_paths(result)
    
    def get_clips_by_threshold(self, query, video_path, similarity_threshold=0.5, max_candidates=100, total_duration=-1):
  
        torch.cuda.empty_cache()
        q_emb = self.calculate_text_embedding(query, flag_save_embedding=False).cpu()
        q_emb = q_emb / q_emb.norm(p=2, dim=1, keepdim=True)
        if video_path not in self.clip_embs_cache:
            if len(self.clip_embs_cache) > 1:  
                self.clip_embs_cache = {}
            video_name = video_path.split('/')[-1].split('.')[0]
            folder_path = f'{self.dataset_folder}/embeddings/{self.clip_duration}/{self.retriever_type}'
            video_clip_paths, clip_embs = self.calculate_video_clip_embedding(video_path, folder_path, total_duration)
            if isinstance(clip_embs, dict):
                clip_embs = clip_embs['video']
            clip_embs = clip_embs.cpu()
            self.clip_embs_cache[video_path] = (video_clip_paths, clip_embs)
        else:
            video_clip_paths, clip_embs = self.clip_embs_cache[video_path]

    
        clip_embs = clip_embs / clip_embs.norm(p=2, dim=1, keepdim=True)

    
        similarities = torch.matmul(q_emb, clip_embs.T)[0]  # shape: [num_clips]


        top_idx = similarities.argsort(descending=True)[:max_candidates].tolist()

        result = []
        thr = float(similarity_threshold)
        for i in top_idx:
            sim_score = float(similarities[i].item())
            if sim_score >= thr:
                result.append((video_clip_paths[i], sim_score))

        torch.cuda.empty_cache()
        
        return parse_and_sort_file_paths(result)

    
    @ torch.no_grad()
    def get_informative_captions(self, query, video_path, top_k=1, total_duration=-1, return_embeddings=False,merge_sentence=False,flag_save_embedding=1):
        # if not os.path.exists(video_path.replace('video','caption').replace('.mp4','.json')) and not os.path.exists(video_path.replace('videos','subtitles').replace('.mp4','_en.json')):
        #     return ''

        q_emb = self.text_retriever.encode(query, batch_size=12, max_length=256)['dense_vecs']
        subtitles_with_time = extract_caption(video_path)
        subtitles = [x[2] for x in subtitles_with_time]

        if flag_save_embedding:
            video_name = video_path.split('/')[-1].split('.')[0]
            os.makedirs(f'{self.dataset_folder}/embeddings/caption',exist_ok=True)
            embedding_path = f'{self.dataset_folder}/embeddings/caption/{video_name}_caption.pkl'
            try:
                subtitle_embeddings = pickle.load(open(embedding_path,'rb')).cpu()
            except Exception as e:
                print(e)
                subtitle_embeddings = self.text_retriever.encode(subtitles, batch_size=12, max_length=256)['dense_vecs']
                if flag_save_embedding:
                    pickle.dump(subtitle_embeddings,open(embedding_path,'wb'))

        similarities = np.dot(q_emb, subtitle_embeddings.T).flatten()  # shape: (832,)
        top_k_indices = np.argsort(similarities)[-top_k:][::-1].tolist()
        return [subtitles_with_time[i] for i in top_k_indices]
    @ torch.no_grad()
    def get_informative_subtitles(self,  video_path,query, top_k=1, total_duration=-1, return_embeddings=False,merge_sentence=False,flag_save_embedding=1):
        # 若为lvb ，用wisper模型逻辑，希望和longvideobench的subtitles的样子是一样的
        
        if not os.path.exists(video_path.replace('videos','subtitles').replace('.mp4','.srt')) and not os.path.exists(video_path.replace('videos','subtitles').replace('.mp4','_en.json')) and not os.path.exists(video_path.replace('video','subtitles').replace('.mp4','.srt')):
            return ''
        q_emb = self.text_retriever.encode(query, batch_size=128, max_length=256)['dense_vecs']
        subtitles_with_time = extract_subtitles(video_path)
        # norm_query = normalize(query)
        # matched = [
        # (start, end, text)
        # for (start, end, text) in subtitles_with_time
        # if norm_query in normalize(text) or normalize(text) in norm_query]
        # if matched:
        #     intervals = [(r[0], r[1]) for r in matched]
        #     subtitles = matched
        #     print(intervals,subtitles)
        #     return intervals#,subtitles
        # top_k=5
        subtitles = [x[2] for x in subtitles_with_time]
        if flag_save_embedding:
            video_name = video_path.split('/')[-1].split('.')[0]
            os.makedirs(f'{self.dataset_folder}/embeddings/subtitle/{self.retriever_type}',exist_ok=True)
            embedding_path = f'{self.dataset_folder}/embeddings/subtitle/{self.retriever_type}/{video_name}_subtitle.pkl'
            try:
                subtitle_embeddings = pickle.load(open(embedding_path,'rb'))
            except Exception as e:
                print(e)
                subtitle_embeddings = self.text_retriever.encode(subtitles, batch_size=12, max_length=256)['dense_vecs']
                if flag_save_embedding:
                    pickle.dump(subtitle_embeddings,open(embedding_path,'wb'))
        similarities = np.dot(q_emb, subtitle_embeddings.T).flatten()  # shape: (832,)
        top_k_indices = np.argsort(similarities)[-top_k:][::-1].tolist()
        # intervals, subtitle
        result=[subtitles_with_time[i] for i in top_k_indices]
        intervals=[(r[0],r[1]) for r in result]
        subtitles=result
        return intervals#,subtitles