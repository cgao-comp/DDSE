import torch
import torch.nn as nn
import torch.nn.functional as F
from ...subNets import BertTextEncoder
from ...subNets.transformers_encoder.transformer import TransformerEncoder
from trains.singleTask.model.TSSA import CausalSelfAttention_TSSA, Config
from models.TC_Mamba import TCSSM

class DDSE(nn.Module):
    def __init__(self, args):
        super(DDSE, self).__init__()
        # BERT文本编码器初始化
        if args.use_bert:
            self.text_model = BertTextEncoder(use_finetune=args.use_finetune, transformers=args.transformers,
                                              pretrained=args.pretrained)
        self.use_bert = args.use_bert
        
        # 特征维度和配置初始化
        dst_feature_dims, nheads = args.dst_feature_dim_nheads
        if args.dataset_name == 'mosi':
            self.len_l, self.len_v, self.len_a = (50, 50, 50) if args.need_data_aligned else (50, 500, 375)
        if args.dataset_name == 'mosei':
            self.len_l, self.len_v, self.len_a = (50, 50, 50) if args.need_data_aligned else (50, 500, 500)
            
        self.orig_d_l, self.orig_d_a, self.orig_d_v = args.feature_dims
        self.d_l = self.d_a = self.d_v = dst_feature_dims
        self.num_heads = nheads
        
        # 初始化各种dropout参数
        self.layers = args.nlevels
        self.attn_dropout = args.attn_dropout
        self.attn_dropout_a = args.attn_dropout_a
        self.attn_dropout_v = args.attn_dropout_v
        self.relu_dropout = args.relu_dropout
        self.embed_dropout = args.embed_dropout
        self.res_dropout = args.res_dropout
        self.output_dropout = args.output_dropout
        self.text_dropout = args.text_dropout
        self.attn_mask = args.attn_mask

        # 特征维度设置
        combined_dim = 2 * self.d_l + self.d_a + self.d_v + self.d_l * 3
        output_dim = 1

        # 初始化时序卷积层
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=args.conv1d_kernel_size_l, padding=0, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=args.conv1d_kernel_size_a, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=args.conv1d_kernel_size_v, padding=0, bias=False)

        # 模态特定编码器
        self.encoder_s_l = nn.Conv1d(self.d_l, self.d_l, kernel_size=1, padding=0, bias=False)
        self.encoder_s_v = nn.Conv1d(self.d_v, self.d_v, kernel_size=1, padding=0, bias=False)
        self.encoder_s_a = nn.Conv1d(self.d_a, self.d_a, kernel_size=1, padding=0, bias=False)

        # 模态不变编码器
        self.encoder_c = nn.Conv1d(self.d_l, self.d_l, kernel_size=1, padding=0, bias=False)

        # 解码器
        self.decoder_l = nn.Conv1d(self.d_l * 2, self.d_l, kernel_size=1, padding=0, bias=False)
        self.decoder_v = nn.Conv1d(self.d_v * 2, self.d_v, kernel_size=1, padding=0, bias=False)
        self.decoder_a = nn.Conv1d(self.d_a * 2, self.d_a, kernel_size=1, padding=0, bias=False)

        # 余弦相似度投影层
        self.proj_cosine_l = nn.Linear(self.d_a * (self.len_l - args.conv1d_kernel_size_l + 1), self.d_a)
        self.proj_cosine_v = nn.Linear(self.d_a * (self.len_v - args.conv1d_kernel_size_v + 1), self.d_a)
        self.proj_cosine_a = nn.Linear(self.d_a * (self.len_a - args.conv1d_kernel_size_a + 1), self.d_a)
        
        # 特征对齐层
        self.align_c_l = nn.Linear(self.d_a * (self.len_l - args.conv1d_kernel_size_l + 1), self.d_a)
        self.align_c_v = nn.Linear(self.d_a * (self.len_v - args.conv1d_kernel_size_v + 1), self.d_a)
        self.align_c_a = nn.Linear(self.d_a * (self.len_a - args.conv1d_kernel_size_a + 1), self.d_a)

        # 初始化TSSA配置和层
        tssa_config_l = Config(n_embd=self.d_l, n_head=self.num_heads, bias=True, dropout=0.1, 
                             block_size=self.len_l-args.conv1d_kernel_size_l+1)
        tssa_config_v = Config(n_embd=self.d_v, n_head=self.num_heads, bias=True, dropout=0.1, 
                             block_size=self.len_v-args.conv1d_kernel_size_v+1)
        tssa_config_a = Config(n_embd=self.d_a, n_head=self.num_heads, bias=True, dropout=0.1, 
                             block_size=self.len_a-args.conv1d_kernel_size_a+1)
        
        self.self_attentions_c_l = CausalSelfAttention_TSSA(tssa_config_l)
        self.self_attentions_c_v = CausalSelfAttention_TSSA(tssa_config_v)
        self.self_attentions_c_a = CausalSelfAttention_TSSA(tssa_config_a)
        self.self_attentions_c_l_l = self.get_network(self_type='l')

        # 投影层和输出层
        self.proj1_c = nn.Linear(self.d_l * 3, self.d_l * 3)
        self.proj2_c = nn.Linear(self.d_l * 3, self.d_l * 3)
        self.out_layer_c = nn.Linear(self.d_l * 3, output_dim)

        # 跨模态注意力层
        self.trans_l_with_a = self.get_network(self_type='la')
        self.trans_l_with_v = self.get_network(self_type='lv')
        self.trans_a_with_l = self.get_network(self_type='al')
        self.trans_a_with_v = self.get_network(self_type='av')
        self.trans_v_with_l = self.get_network(self_type='vl')
        self.trans_v_with_a = self.get_network(self_type='va')
        self.trans_l_mem = self.get_network(self_type='l_mem', layers=3)
        self.trans_a_mem = self.get_network(self_type='a_mem', layers=3)
        self.trans_v_mem = self.get_network(self_type='v_mem', layers=3)

        # 模态权重层和最终输出层
        self.weight_l = nn.Linear(2 * self.d_l, 2 * self.d_l)
        self.weight_v = nn.Linear(self.d_v, self.d_v)
        self.weight_a = nn.Linear(self.d_a, self.d_a)
        self.weight_c = nn.Linear(3 * self.d_l, 3 * self.d_l)
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)

        
        # TCSSM模块
        self.tcssm = TCSSM(
            num_layers=8,
            input_size=self.d_l,
            output_sizes=[self.d_l*2, self.d_l],
            d_ffn=1024,
            activation='Swish',
            dropout=0.4,
            mamba_config={
                'd_state': 16,
                'd_conv': 4,
                'expand': 2,
                'bidirectional': True
            }
        )

        # 最终特征转换层
        self.final_layer_l = nn.Linear(self.d_l*2, 1)
        self.final_layer_v = nn.Linear(self.d_l, 1)
        self.final_layer_a = nn.Linear(self.d_l, 1)

    def get_network(self, self_type='l', layers=-1):
        # 根据模态类型选择网络参数
        if self_type in ['al', 'vl']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type in ['l']:
            embed_dim, attn_dropout = 2*self.d_l, self.attn_dropout
        elif self_type in ['a', 'la', 'va']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ['v', 'lv', 'av']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        elif self_type == 'l_mem':
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = self.d_a, self.attn_dropout
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = self.d_v, self.attn_dropout
        else:
            raise ValueError("Unknown network type")

        return TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=self.num_heads,
            layers=max(self.layers, layers),
            attn_dropout=attn_dropout,
            relu_dropout=self.relu_dropout,
            res_dropout=self.res_dropout,
            embed_dropout=self.embed_dropout,
            attn_mask=self.attn_mask
        )

    def forward(self, text, audio, video):
        # BERT文本处理
        if self.use_bert:
            text = self.text_model(text)
        
        # 特征预处理
        x_l = F.dropout(text.transpose(1, 2), p=self.text_dropout, training=self.training)
        x_a = audio.transpose(1, 2)
        x_v = video.transpose(1, 2)

        # 特征投影
        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)

        # 特定模态和不变模态编码
        s_l = self.encoder_s_l(proj_x_l)
        s_v = self.encoder_s_v(proj_x_v)
        s_a = self.encoder_s_a(proj_x_a)

        c_l = self.encoder_c(proj_x_l)
        c_v = self.encoder_c(proj_x_v)
        c_a = self.encoder_c(proj_x_a)
        c_list = [c_l, c_v, c_a]

        # 特征对齐
        c_l_sim = self.align_c_l(c_l.contiguous().view(x_l.size(0), -1))
        c_v_sim = self.align_c_v(c_v.contiguous().view(x_l.size(0), -1))
        c_a_sim = self.align_c_a(c_a.contiguous().view(x_l.size(0), -1))

        # 特征重构
        recon_l = self.decoder_l(torch.cat([s_l, c_list[0]], dim=1))
        recon_v = self.decoder_v(torch.cat([s_v, c_list[1]], dim=1))
        recon_a = self.decoder_a(torch.cat([s_a, c_list[2]], dim=1))

        s_l_r = self.encoder_s_l(recon_l)
        s_v_r = self.encoder_s_v(recon_v)
        s_a_r = self.encoder_s_a(recon_a)

        # 维度调整和注意力处理
        s_l = s_l.permute(2, 0, 1)
        s_v = s_v.permute(2, 0, 1)
        s_a = s_a.permute(2, 0, 1)

        c_l = c_l.transpose(1, 2)
        c_v = c_v.transpose(1, 2)
        c_a = c_a.transpose(1, 2)

        # TSSA自注意力处理
        c_l_att = self.self_attentions_c_l(c_l)
        if type(c_l_att) == tuple:
            c_l_att = c_l_att[0]
        c_l_att = c_l_att.transpose(0, 1)[-1]

        c_v_att = self.self_attentions_c_v(c_v)
        if type(c_v_att) == tuple:
            c_v_att = c_v_att[0]
        c_v_att = c_v_att.transpose(0, 1)[-1]

        c_a_att = self.self_attentions_c_a(c_a)
        if type(c_a_att) == tuple:
            c_a_att = c_a_att[0]
        c_a_att = c_a_att.transpose(0, 1)[-1]

        # 特征融合
        c_fusion = torch.cat([c_l_att, c_v_att, c_a_att], dim=1)
        c_proj = self.proj2_c(F.dropout(F.relu(self.proj1_c(c_fusion), inplace=True), 
                                      p=self.output_dropout, training=self.training))
        c_proj += c_fusion
        logits_c = self.out_layer_c(c_proj)

        # 模态间交互
        h_a_with_v = self.trans_a_with_v(s_a, s_v, s_v)
        h_v_with_a = self.trans_v_with_a(s_v, s_a, s_a)
        
        h_a = self.trans_a_mem(h_a_with_v)
        h_v = self.trans_v_mem(h_v_with_a)
        
        if type(h_a) == tuple:
            h_a = h_a[0]
        if type(h_v) == tuple:
            h_v = h_v[0]

        # TCSSM处理
        s_l = s_l.permute(1, 0, 2)
        s_v = h_v.permute(1, 0, 2)
        s_a = h_a.permute(1, 0, 2)

        l_a_out, a_l_out = self.tcssm(s_l, s_a)
        l_v_out, v_l_out = self.tcssm(s_l, s_v)

        # 特征合并和转换
        h_ls = torch.cat([l_a_out, l_v_out], dim=2)
        h_ls = h_ls.permute(1, 0, 2)
        h_ls = self.self_attentions_c_l_l(h_ls)
        if type(h_ls) == tuple:
            h_ls = h_ls[0]
        last_h_l = last_hs = h_ls[-1]

        h_as = a_l_out.permute(1, 0, 2)
        if type(h_as) == tuple:
            h_as = h_as[0]
        last_h_a = last_hs = h_as[-1]

        h_vs = v_l_out.permute(1, 0, 2)
        if type(h_vs) == tuple:
            h_vs = h_vs[0]
        last_h_v = last_hs = h_vs[-1]

        # 最终特征转换
        last_last_h_l = self.final_layer_l(last_h_l)
        last_last_h_v = self.final_layer_v(last_h_v)
        last_last_h_a = self.final_layer_a(last_h_a)

        # 特征加权和融合
        last_h_l = torch.sigmoid(self.weight_l(last_h_l))
        last_h_v = torch.sigmoid(self.weight_v(last_h_v))
        last_h_a = torch.sigmoid(self.weight_a(last_h_a))
        c_fusion = torch.sigmoid(self.weight_c(c_fusion))

        # 最终输出
        last_hs = torch.cat([last_h_l, last_h_v, last_h_a, c_fusion], dim=1)
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs), inplace=True), 
                                          p=self.output_dropout, training=self.training))
        last_hs_proj += last_hs
        output = self.out_layer(last_hs_proj)

        # 维度调整
        s_l = s_l.permute(1, 0, 2)
        s_v = s_v.permute(1, 0, 2)
        s_a = s_a.permute(1, 0, 2)

        # 返回结果字典
        return {
            'origin_l': proj_x_l,
            'origin_v': proj_x_v,
            'origin_a': proj_x_a,
            's_l': s_l,
            's_v': s_v,
            's_a': s_a,
            'c_l': c_l,
            'c_v': c_v,
            'c_a': c_a,
            's_l_r': s_l_r,
            's_v_r': s_v_r,
            's_a_r': s_a_r,
            'recon_l': recon_l,
            'recon_v': recon_v,
            'recon_a': recon_a,
            'c_l_sim': c_l_sim,
            'c_v_sim': c_v_sim,
            'c_a_sim': c_a_sim,
            'last_h_l': last_last_h_l,
            'last_h_v': last_last_h_v,
            'last_h_a': last_last_h_a,
            'logits_c': logits_c,
            'output_logit': output
        }