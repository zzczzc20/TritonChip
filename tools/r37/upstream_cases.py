"""Small numerical cases for TileBench's ORIGINAL JIT functions.

Only launch metadata, addresses and input data are adapted to the RTL.
The loader reads unchanged function bodies from the specified checkout.
"""
import math
from pathlib import Path

import cases as C
import upstream as U


def build(Case, source):
    root = Path(source) / 'benchmarks/operators'
    if not root.exists():
        root = Path(source) / 'tilebench/benchmarks/operators'
    result = {}

    def add(oid, fn_name, buffers, scalars, output, expect, *, ce=None,
            types=None, label='', **kwargs):
        fn = U.load(root / oid / 'impl_triton.py')[fn_name]
        sig, constexprs, _ = U.configuration(oid, fn)
        constexprs.update(ce or {})
        for name, (dtype, _) in buffers.items():
            if name in sig:
                sig[name] = '*' + dtype
        sig.update(types or {})
        case = Case(oid, kernel_spec=(fn, sig, constexprs), buffers=buffers,
                    scalars=scalars, out=output, expect=expect, **kwargs)
        result[oid + ('/' + label if label else '')] = case
        return case

    n = 13
    x = [float(i - 6) * 0.25 for i in range(16)]
    y = [float((i * 7) % 13 - 6) * 0.125 for i in range(16)]
    for oid, fn, out, expect in (
        ('mul2', 'mul2_kernel', 'output_ptr', [v * 2 for v in x]),
        ('relu', 'relu_kernel', 'output_ptr', [max(v, 0) for v in x]),
        ('leaky_relu', 'leaky_relu_kernel', 'y_ptr', [v if v > 0 else C.f32(v * C.f32(.01)) for v in x]),
        ('sigmoid', 'sigmoid_kernel', 'y_ptr', [1/(1+math.exp(-v)) for v in x]),
        ('quantize_global', 'quantize_kernel', 'out_ptr', [C.f16(v) for v in x]),
    ):
        add(oid, fn, {'x_ptr': ('fp32', x), out: ('fp16' if oid == 'quantize_global' else 'fp32', [0]*16)},
            {'n_elements': n}, out, expect[:n], rtol=C.TOL_EXP if oid == 'sigmoid' else 0)
    add('vector_add', 'add_kernel', {'x_ptr': ('fp32', x), 'y_ptr': ('fp32', y),
        'output_ptr': ('fp32', [0]*16)}, {'n_elements': n}, 'output_ptr', [a+b for a,b in zip(x[:n],y[:n])])
    add('swiglu', 'swiglu_kernel', {'x_ptr': ('fp32', x), 'y_ptr': ('fp32', y),
        'out_ptr': ('fp32', [0]*16)}, {'n_elements': n}, 'out_ptr',
        [a/(1+math.exp(-a))*b for a,b in zip(x[:n],y[:n])], rtol=C.TOL_EXP)
    bias = [i*.125 for i in range(16)]
    z = [a*b+c for a,b,c in zip(x,y,bias)]
    add('fused_activation', 'fused_activation_kernel', {'x_ptr': ('fp32', x), 'gate_ptr': ('fp32',y),
        'bias_ptr': ('fp32',bias), 'out_ptr': ('fp32',[0]*16)}, {'n_elements':n}, 'out_ptr',
        [v/(1+math.exp(-v)) for v in z[:n]],rtol=C.TOL_EXP)
    keep = [int(i%3!=0) for i in range(16)]
    add('dropout','dropout_kernel',{'x_ptr':('fp32',x),'x_keep_ptr':('i32',keep),
        'output_ptr':('fp32',[0]*16)}, {'n_elements':n,'p':.5}, 'output_ptr',
        [a*2 if k else 0 for a,k in zip(x[:n],keep[:n])])
    q = [i*7-60 for i in range(16)]
    add('dequantize_rowwise','dequantize_rowwise_kernel', {'x_ptr':('i8',q),'state_ptr':('fp32',[.5]),
        'out_ptr':('fp16',[0]*16)}, {'ROWS':1,'COLS':n}, 'out_ptr',
        [C.f16(C.f32(v*.5*C.f32(1/127))) for v in q[:n]])
    add('weight_dequant','dequant_kernel',{'X':('fp32',x),'S':('fp32',[.5,.25,.125,.0625]),
        'Y':('fp32',[0]*16)}, {}, 'Y', [x[i]*[.5,.25,.125,.0625][i//4] for i in range(16)],
        ce={'M':1,'N':16,'TILE_SIZE':4,'S_COLS':4})

    for n in (7, 37):
        vals = [float((i*7)%19-9)*.125 for i in range(n)]
        w = [1+i*.125 for i in range(n)]
        bias = [i*.125 for i in range(n)]
        p = [math.exp(v-max(vals)) for v in vals]
        add('softmax','softmax_online_kernel',{'input_ptr':('fp32',vals),'output_ptr':('fp32',[0]*n)},
            {'input_row_stride':n,'output_row_stride':n,'n_cols':n}, 'output_ptr', [v/sum(p) for v in p],
            label=f'n{n}',rtol=C.TOL_CHAIN)
        mu=sum(vals)/n
        scale=math.sqrt(sum((v-mu)**2 for v in vals)/n+.125)
        add('layernorm','layernorm_kernel',{'x_ptr':('fp32',vals),'weight_ptr':('fp32',w),
            'bias_ptr':('fp32',bias),'out_ptr':('fp32',[0]*n)}, {'stride_row':n,'N_SIZE':n,'eps':.125},
            'out_ptr',[(v-mu)/scale*ww+b for v,ww,b in zip(vals,w,bias)],label=f'n{n}',rtol=C.TOL_CHAIN,atol=2**-18)
        scale=math.sqrt(sum(v*v for v in vals)/n+.125)
        add('rmsnorm','rmsnorm_kernel',{'x_ptr':('fp32',vals),'rms_w_ptr':('fp32',w),'out_ptr':('fp32',[0]*n)},
            {'stride_row':n,'N_SIZE':n,'eps':.125},'out_ptr',[v/scale*ww for v,ww in zip(vals,w)],
            label=f'n{n}',rtol=C.TOL_CHAIN)
        scale=math.sqrt(sum(v*v for v in vals)+.125)
        add('l2_norm','_l2_norm_fwd_kernel',{'X':('fp32',vals),'Y':('fp32',[0]*n)},
            {'stride_x_row':n,'eps':.125},'Y',[v/scale for v in vals],ce={'N':n},label=f'n{n}',rtol=C.TOL_CHAIN)
        pp=[(i%7)*.125 for i in range(n)]
        add('kl_divergence','kl_divergence_kernel',{'log_y_pred_ptr':('fp32',vals),'y_true_ptr':('fp32',pp),
            'loss_ptr':('fp32',[0])}, {'log_y_pred_stride':n,'y_true_stride':n,'n_cols':n}, 'loss_ptr',
            [sum(p*(math.log(p)-v) if p else 0 for p,v in zip(pp,vals))], label=f'n{n}',rtol=C.TOL_CHAIN)
        add('argmax','argmax_rowwise_kernel',{'X':('fp32',vals),'Out':('i64',[0])}, {'N':n},'Out',
            [vals.index(max(vals))],label=f'n{n}')
    add('mean_reduction','mean_rowwise_kernel',{'X':('fp32',x),'Out':('fp32',[0])}, {'M':1},'Out',
        [sum(x)/16],ce={'N':16,'BLOCK_M':1})
    add('cross_entropy','cross_entropy_kernel',{'logits_ptr':('fp32',x),'targets_ptr':('i32',[5]),
        'output_ptr':('fp32',[0])},{'num_classes':16,'stride_bn':16,'stride_bc':1},'output_ptr',
        [max(x)+math.log(sum(math.exp(v-max(x)) for v in x))-x[5]],rtol=C.TOL_CHAIN)
    top=sorted(range(16),key=lambda i:(-x[i],i))[:4]
    ex=[math.exp(x[i]-x[top[0]]) for i in top]
    add('moe_topk_gating','moe_topk_gating_kernel',{'logits_ptr':('fp32',x),'topk_w_ptr':('fp32',[0]*4),
        'topk_idx_ptr':('i32',[0]*4)}, {'E':16},'topk_w_ptr',[v/sum(ex) for v in ex],
        ce={'K':4,'BLOCK_SIZE_K':4,'BLOCK_SIZE_E':16},extra_outputs=[('topk_idx_ptr',0,top)],rtol=C.TOL_CHAIN)
    add('top_k_selection','block_topk_kernel',{'input_ptr':('fp32',y),'out_ptr':('fp32',[0]*4)},
        {'n':13},'out_ptr', sorted(y[:13],reverse=True)[:4],ce={'K2':4})
    add('interleave','interleave_kernel',{'A_ptr':('fp32',x),'B_ptr':('fp32',y),'output_ptr':('fp32',[0]*16)},
        {'N':7},'output_ptr',[v for pair in zip(x[:7],y[:7]) for v in pair],ce={'BLOCK_SIZE':8})
    add('reverse_array','reverse_kernel',{'input_ptr':('fp32',x),'output_ptr':('fp32',[0]*16)},
        {'N':13},'output_ptr',list(reversed(x[:13])))
    add('matrix_copy','matrix_copy_kernel',{'A_ptr':('fp32',x),'B_ptr':('fp32',[0]*16)},
        {'N':4},'B_ptr',x)
    a=[float((i*7)%17-8)*.25 for i in range(16*16)]
    add('matrix_transpose','transpose_kernel',{'x_ptr':('fp32',a),'output_ptr':('fp32',[0]*256)},
        {'m':16,'n':16,'stride_xm':16,'stride_xn':1,'stride_om':16,'stride_on':1},
        'output_ptr',[a[j*16+i] for i in range(16) for j in range(16)])
    # In-place rotary embedding: two heads, masked half-head tail.
    q=[(i-13)*.125 for i in range(32)]
    sn=[.25]*8; cs=[.5]*8
    expected=[]
    for h in range(2):
        lo=q[h*16:h*16+8]; hi=q[h*16+8:h*16+16]
        expected += [a*.5-b*.25 for a,b in zip(lo,hi)]
        expected += [b*.5+a*.25 for a,b in zip(lo,hi)]
    add('rope','rope_embedding',{'Q':('fp32',q),'cos':('fp32',cs),'sin':('fp32',sn)},
        {'Q_row_stride':32,'cos_row_stride':8,'sin_row_stride':8,'seqlen':1},'Q',expected,
        ce={'head_dim':16,'n_heads':2,'ROPE_GROUP_SIZE':2},preserve_output=True)
    dest=[2,0,3,1]
    expected=[0]*16
    for i in range(16): expected[dest[i//4]*4+i%4]=x[i]
    add('destindex','copy_by_dest_kernel',{'kv_ptr':('fp32',x),'dest_ptr':('i32',dest),
        'out_ptr':('fp32',[0]*16)}, {'total':16},'out_ptr',expected,ce={'head_num':2,'head_dim':2})
    h,w=3,13
    a=[((i*7)%23-11)*.125 for i in range(h*w)]
    weights=[v/16 for v in (1,2,1,2,4,2,1,2,1)]
    expected=[sum(a[rr*w+cc]*weights[kr*3+kc]
        for kr in range(3) for kc in range(3)
        for rr,cc in [(r+kr-1,c+kc-1)] if 0<=rr<h and 0<=cc<w)
        for r in range(h) for c in range(w)]
    add('gaussian_blur','gaussian_blur_kernel',{'input_ptr':('fp32',a),
        'kernel_ptr':('fp32',weights),'output_ptr':('fp32',[0]*(h*w))},
        {'input_rows':h,'input_cols':w},'output_ptr',expected,ce={'BLOCK_R':4})
    expected=[max(a[rr*w+cc] for rr in range(r-1,r+2) for cc in range(c-1,c+2)
                  if 0<=rr<h and 0<=cc<w) for r in range(h) for c in range(w)]
    add('2d_max_pooling','max_pool2d_kernel',{'input_ptr':('fp32',a),'output_ptr':('fp32',[0]*(h*w))},
        {'H':h,'W':w,'H_out':h,'W_out':w},'output_ptr',expected,ce={'BLOCK_R':4,'padding':1})
    expected=[a[r*w+c] if r in (0,h-1) or c in (0,w-1) else
        .25*(a[(r-1)*w+c]+a[(r+1)*w+c]+a[r*w+c-1]+a[r*w+c+1])
        for r in range(h) for c in range(w)]
    add('jacobi_stencil_2d','jacobi_stencil_kernel',{'input':('fp32',a),'output':('fp32',[0]*(h*w))},
        {'rows':h,'cols':w,'stride_ir':w,'stride_ic':1,'stride_or':w,'stride_oc':1},
        'output',expected,ce={'BLOCK_SIZE_R':4})
    for dtype in ('fp16','fp32','e4m3','e5m2'):
        a=[float((i*7)%11-5)*.25 for i in range(16*32)]
        bt=[float((i*3)%9-4)*.25 for i in range(16*32)]
        expected=[sum(a[m*32+k]*bt[n*32+k] for k in range(32))
                  for m in range(16) for n in range(16)]
        if dtype.startswith('e'):
            from execute import DTYPES
            _, encode, decode = DTYPES[dtype]
            expected = [decode(encode(v)) for v in expected]
        desc={name:{'buffer':buf,'shape':[16,k],'strides':[k,1],'block':[16,16]}
              for name,buf,k in [('a_desc','a',32),('b_desc','bt',32),('c_desc','c',16)]}
        add('matmul_fp32_fp16_fp8','matmul_kernel',{'a':(dtype,a),'bt':(dtype,bt),'c':(dtype,[0]*256)},
            {'M':16,'N':16,'K':32},'c',expected,label=dtype,descriptors=desc,
            types={name:f'tensordesc<{dict(e4m3="fp8e4nv",e5m2="fp8e5").get(dtype,dtype)}[16,16]>' for name in desc})
    a=[(i*7)%17-8 for i in range(16*64)]
    packed=[(i*37+13)%256 for i in range(16*16)]
    expected=[sum(a[m*64+k]*(((packed[n*16+k%16] >> (2*(k//16))) & 3)-1)
                  for k in range(64)) for m in range(16) for n in range(16)]
    desc={name:{'buffer':buf,'shape':[16,k],'strides':[k,1],'block':[16,16]}
          for name,buf,k in [('a_desc','a',64),('b_desc','bt',16),('c_desc','c',16)]}
    add('matmul_int8','matmul_kernel',{'a':('i8',a),'bt':('i8',[v if v<128 else v-256 for v in packed]),
        'c':('i32',[0]*256)}, {'M':16,'N':16},'c',expected,descriptors=desc)
    a=[((i*7)%11-5)*.25 for i in range(2*256)]
    bt=[((i*3)%9-4)*.25 for i in range(2*256)]
    expected=[sum(a[256+m*16+k]*bt[256+n*16+k] for k in range(16))
              for m in range(16) for n in range(16)]
    desc={name:{'buffer':buf,'shape':[2,16,16],'strides':[256,16,1],'block':[1,16,16]}
          for name,buf in [('a_desc','a'),('b_desc','bt'),('c_desc','c')]}
    add('batched_matmul','bmm_kernel',{'a':('fp16',a),'bt':('fp16',bt),'c':('fp16',[0]*512)},
        {'M':16,'N':16,'K':16},'c',expected,descriptors=desc,out_offset=256,program_id=(0,0,1))
    d=13; mid=[((i*7)%19-9)*.125 for i in range(3*d)]; logs=[-.5,.25,.75]
    probs=[math.exp(v-max(logs)) for v in logs]
    add('flash_decode','fwd_kernel_flash_decode_stage2',{'B_Seqlen':('i32',[37]),'Mid_O':('fp32',mid),
        'Mid_O_LogExpSum':('fp32',logs),'Out':('fp32',[0]*d)},
        {'stride_mid_ob':3*d,'stride_mid_oh':3*d,'stride_mid_os':d,'stride_mid_od':1,
         'stride_mid_o_eb':3,'stride_mid_o_eh':3,'stride_mid_o_es':1,'stride_obs':d,'stride_oh':d,
         'stride_od':1,'head_dim':d},'Out',[sum(probs[b]*mid[b*d+i] for b in range(3))/sum(probs)
                                         for i in range(d)],rtol=C.TOL_CHAIN,atol=2**-20)
    # Every stage consumes the preceding RTL dump, never a host replacement.
    vals=[.5,-1,.25,.75,-.5,1.25,-.25]
    work=vals+[math.inf]
    add('bitonic_sort','pad_kernel',{'data_ptr':('fp32',vals),'work_ptr':('fp32',[0]*8)},
        {'N':7,'M':8},'work_ptr',work,label='pad')
    previous='bitonic_sort/pad'
    for k in (2,4,8):
        j=k//2
        while j:
            ref=work.copy()
            for i in range(8):
                pair=i^j
                if pair>i:
                    lo,hi=sorted((work[i],work[pair]))
                    ref[i],ref[pair]=(lo,hi) if i&k==0 else (hi,lo)
            label=f'k{k}j{j}'
            add('bitonic_sort','bitonic_step_kernel',{'work_ptr':('fp32',[0]*8)},
                {'k':k,'j':j,'M':8},'work_ptr',ref,label=label,preserve_output=True,
                feeds={'work_ptr':(previous,'work_ptr')})
            previous='bitonic_sort/'+label
            work=ref; j//=2
    assert work==sorted(vals)+[math.inf]
    vals=[(i-4)*.25 for i in range(9)]
    sm=sum(vals); ss=sum(v*v for v in vals); mu=sm/len(vals)
    inv=1/math.sqrt(ss/len(vals)-mu*mu+.125)
    add('batch_normalization','compute_block_sums_kernel',{'input_ptr':('fp32',vals),
        'block_sum_ptr':('fp32',[0]),'block_sq_sum_ptr':('fp32',[0])},{'N':9},'block_sum_ptr',[sm],
        label='sums',ce={'C':1,'C_P2':1,'BLOCK_N':16,'STEP':4},extra_outputs=[('block_sq_sum_ptr',0,[ss])])
    add('batch_normalization','compute_mean_invstd_kernel',{'block_sum_ptr':('fp32',[0]),
        'block_sq_sum_ptr':('fp32',[0]),'mean_ptr':('fp32',[0]),'inv_std_ptr':('fp32',[0])},
        {'N':9,'C':1,'NUM_BLOCKS':1,'eps':.125},'inv_std_ptr',[inv],label='stats',
        extra_outputs=[('mean_ptr',0,[mu])],rtol=C.TOL_CHAIN,
        feeds={b:('batch_normalization/sums',b) for b in ('block_sum_ptr','block_sq_sum_ptr')})
    add('batch_normalization','apply_batch_norm_kernel',{'input_ptr':('fp32',vals),
        'gamma_ptr':('fp32',[.5]),'beta_ptr':('fp32',[.25]),'output_ptr':('fp32',[0]*9),
        'mean_ptr':('fp32',[0]),'inv_std_ptr':('fp32',[0])},{'N':9},'output_ptr',
        [(v-mu)*inv*.5+.25 for v in vals],label='apply',ce={'C':1,'C_P2':1,'ROWS':16,'STEP':4},
        rtol=C.TOL_CHAIN,atol=2**-20,feeds={b:('batch_normalization/stats',b) for b in ('mean_ptr','inv_std_ptr')})
    vals=[0,1,1,3,-1,7,2,1,0,8,3,2,1,7,0,0,2,2,3]
    hist=[vals.count(i) for i in range(8)]
    add('histogramming','histogram_partial_kernel',{'input_ptr':('i32',vals),'partial_ptr':('i32',[0]*8)},
        {'N':len(vals),'num_bins':8,'num_partials':1,'stride_pr':8,'stride_pb':1},
        'partial_ptr',hist,label='partial',preserve_output=True)
    add('histogramming','histogram_reduce_kernel',{'partial_ptr':('i32',[0]*8),'hist_ptr':('i32',[0]*8)},
        {'num_partials':1,'num_bins':8,'stride_pr':8,'stride_pb':1},'hist_ptr',hist,label='reduce',
        ce={'BLOCK_ROWS':1},feeds={'partial_ptr':('histogramming/partial','partial_ptr')})
    import itertools
    for rank, spatial, kernel, pad in [(1,(7,),(3,),1),(2,(3,4),(3,3),1),(3,(3,3,3),(2,2,2),0)]:
        axes={1:'l',2:'hw',3:'dhw'}[rank]
        out_shape=tuple(n+2*pad-k+1 for n,k in zip(spatial,kernel))
        ni,nk,no=math.prod(spatial),math.prod(kernel),math.prod(out_shape)
        data=[((i*3)%11-5)*.25 for i in range(ni)]
        weights=[((i*7)%9-4)*.125 for i in range(2*nk)]
        def flat(at,shape):
            value=0
            for i,n in zip(at,shape): value=value*n+i
            return value
        expected=[]
        for oc in range(2):
            for loc in itertools.product(*(range(n) for n in out_shape)):
                total=0
                for tap in itertools.product(*(range(n) for n in kernel)):
                    at=tuple(i+j-pad for i,j in zip(loc,tap))
                    if all(0<=i<n for i,n in zip(at,spatial)):
                        total+=data[flat(at,spatial)]*weights[oc*nk+flat(tap,kernel)]
                expected.append(C.f16(total))
        args={'batch':1,'in_channels':1,'out_channels':2,'groups':1,
              'out_channels_per_group':2,'in_channels_per_group':1,
              'stride_input_b':ni,'stride_input_c':ni,'stride_weight_oc':nk,
              'stride_weight_ic':nk,'stride_output_b':2*no,'stride_output_c':no}
        for i,axis in enumerate(axes):
            args.update({f'in_{axis.upper()}':spatial[i],f'out_{axis.upper()}':out_shape[i],
                f'k{axis.upper()}':kernel[i],f'stride_{axis}':1,f'pad_{axis}':pad,
                f'stride_input_{axis}':math.prod(spatial[i+1:]),
                f'stride_weight_k{axis}':math.prod(kernel[i+1:]),
                f'stride_output_{axis}':math.prod(out_shape[i+1:])})
        add(f'{rank}d_conv',f'conv{rank}d_kernel',{'input_ptr':('fp16',data),
            'weight_ptr':('fp16',weights),'output_ptr':('fp16',[0]*(2*no))},args,'output_ptr',expected)
    q=[((i*7)%11-5)*.125 for i in range(256)]
    k=[((i*3)%13-6)*.125 for i in range(256)]
    v=[((i*5)%17-8)*.125 for i in range(256)]
    for oid in ('flash_attention','block_sparse_attention'):
        scale=.125
        expected=[]
        for m in range(16):
            if oid=='flash_attention':
                qs=[C.f16(C.f32(x*C.f32(scale*C.f32(1.44269504)))) for x in q[m*16:(m+1)*16]]
                scores=[sum(qs[d]*k[n*16+d] for d in range(16)) for n in range(m+1)]
                probs=[2**(x-max(scores)) for x in scores]
            else:
                scores=[sum(q[m*16+d]*k[n*16+d] for d in range(16))*scale for n in range(m+1)]
                probs=[math.exp(x-max(scores)) for x in scores]
            expected += [C.f16(sum(C.f16(probs[n])*v[n*16+d] for n in range(m+1))/sum(probs))
                         for d in range(16)]
        buffers={'q':('fp16',q),'k':('fp16',k),'v':('fp16',v),'o':('fp16',[0]*256)}
        names=(('desc_q','q'),('desc_k','k'),('desc_v','v'),('desc_o','o')) if oid=='flash_attention' else (
            ('q_desc','q'),('k_desc','k'),('v_desc','v'),('out_desc','o'))
        desc={name:{'buffer':buf,'shape':[1,1,16,16],'strides':[256,256,16,1],'block':[1,1,16,16]}
              for name,buf in names}
        if oid=='flash_attention':
            args={'sm_scale':scale,'BS':1,'HEAD':1,'SEQLEN':16}; ce={'IS_CAUSAL':True}
            fn='fwd_kernel'
        else:
            buffers.update({'layout_csr_row_indices':('i32',[0,1]),'layout_csr_col_indices':('i32',[0])})
            args={'layout_csr_row_stride_h':2,'layout_csr_col_stride_h':1}
            ce={'softmax_scale':scale,'total_seq_len':16}; fn='block_sparse_attention_kernel'
        add(oid,fn,buffers,args,'o',expected,ce=ce,descriptors=desc,rtol=2**-10,atol=2**-13)
    values=[0x70000001,3,0x40000211,0,0x7fffffff,17,3]
    previous=None
    for shift in range(0,32,2):
        prefix=f's{shift}'
        counts=[sum(((v>>shift)&3)==d for v in values) for d in range(4)]
        offsets=[sum(counts[:d]) for d in range(4)]
        feeds={'input':(previous,'output')} if previous else {}
        add('radix_sort','radix_histogram_kernel',{'input':('i32',values),'hist':('i32',[0]*4)},
            {'N':7,'K':1,'shift':shift},'hist',counts,label=prefix+'_hist',feeds=feeds)
        add('radix_sort','radix_sum_chunks_kernel',{'src':('i32',[0]*4),'dst':('i32',[0])},
            {'M':4},'dst',[7],label=prefix+'_sum',feeds={'src':('radix_sort/'+prefix+'_hist','hist')})
        add('radix_sort','radix_scan_chunk_sums_kernel',{'sums':('i32',[0])},{'L':1},'sums',[0],
            label=prefix+'_scan_sums',preserve_output=True,ce={'BLOCK_BB':1},
            feeds={'sums':('radix_sort/'+prefix+'_sum','dst')})
        add('radix_sort','radix_scan_chunks_kernel',{'src':('i32',[0]*4),'chunk_offsets':('i32',[0])},
            {'M':4},'src',offsets,label=prefix+'_scan',preserve_output=True,
            feeds={'src':('radix_sort/'+prefix+'_hist','hist'),
                   'chunk_offsets':('radix_sort/'+prefix+'_scan_sums','sums')})
        expected=sorted(values,key=lambda v:(v>>shift)&3)
        feeds.update({'hist':('radix_sort/'+prefix+'_scan','src')})
        add('radix_sort','radix_scatter_kernel',{'input':('i32',values),'output':('i32',[0]*7),'hist':('i32',[0]*4)},
            {'N':7,'K':1,'shift':shift},'output',expected,label=prefix+'_scatter',feeds=feeds)
        previous='radix_sort/'+prefix+'_scatter';values=expected
    assert values==sorted(values)
    q=[(i-8)*.125 for i in range(16)]
    k=[((i*7)%13-6)*.125 for i in range(16)]
    v=[((i*3)%11-5)*.25 for i in range(16)]
    phi=lambda xs:[C.f32(x+1 if x>0 else math.exp(x)) for x in xs]
    pq,pk=phi(q),phi(k)
    for name,data,expected in [('q',q,pq),('k',k,pk)]:
        add('linear_self_attention','phi_kernel',{'X':('fp32',data),'Y':('fp32',[0]*16)},
            {'n_elements':16},'Y',expected,label='phi_'+name,rtol=C.TOL_EXP)
    def desc2(buf,nr,nc):
        return {'buffer':buf,'shape':[nr,nc],'strides':[nc,1],'block':[16,16]}
    ss=[C.f32(a*b) for a in pk for b in v]
    add('linear_self_attention','kv_gemm_kernel',{'kt':('fp32',[0]*16),'vt':('fp32',v),'s':('fp32',[0]*256)},
        {'M':1,'D':16},'s',ss,label='kv',descriptors={'kt_desc':desc2('kt',16,1),
            'vt_desc':desc2('vt',16,1),'s_desc':desc2('s',16,16)},
        feeds={'kt':('linear_self_attention/phi_k','Y')},rtol=C.TOL_CHAIN,
        types={n:'tensordesc<fp32[16,16]>' for n in ('kt_desc','vt_desc','s_desc')})
    add('linear_self_attention','z_kernel',{'PhiK':('fp32',[0]*16),'Z':('fp32',[0]*16)},
        {'M':1,'D':16,'stride_km':16,'stride_kd':1},'Z',pk,label='z',
        feeds={'PhiK':('linear_self_attention/phi_k','Y')},rtol=C.TOL_CHAIN)
    denom=sum(a*b for a,b in zip(pq,pk))+.125
    expected=[C.f32(sum(pq[d]*ss[d*16+i] for d in range(16))/denom) for i in range(16)]
    add('linear_self_attention','out_gemm_kernel',{'q':('fp32',[0]*16),'st':('fp32',[0]*256),
        'o':('fp32',[0]*16),'Z':('fp32',[0]*16)}, {'M':1,'D':16},'o',expected,label='out',
        descriptors={'q_desc':desc2('q',1,16),'st_desc':desc2('st',16,16),'o_desc':desc2('o',1,16)},
        feeds={'q':('linear_self_attention/phi_q','Y'),'st':('linear_self_attention/kv','s',(16,16)),
               'Z':('linear_self_attention/z','Z')},rtol=C.TOL_CHAIN,atol=2**-20,
        types={n:'tensordesc<fp32[16,16]>' for n in ('q_desc','st_desc','o_desc')})
    # Three output tiles: the first wave owns tile zero, full_tiles owns
    # the remaining two. Launch one of each plus its early-return path.
    a=[((i*7)%11-5)*.25 for i in range(48*16)]
    bt=[((i*3)%9-4)*.25 for i in range(256)]
    expected=[sum(a[m*16+k]*bt[n*16+k] for k in range(16)) for m in range(48) for n in range(16)]
    desc={'a_desc':desc2('a',48,16),'b_desc':desc2('bt',16,16)}
    add('streamk_matmul','first_wave',{'a':('fp16',a),'bt':('fp16',bt),'C':('fp32',[0]*768)},
        {'M':48,'N':16,'K':16,'stride_cm':16,'stride_cn':1},'C',expected[:256],label='first',
        descriptors=desc,preserve_output=True)
    for pid in (0,1):
        add('streamk_matmul','full_tiles',{'a':('fp16',a),'bt':('fp16',bt),'c':('fp32',[0]*768)},
            {'M':48,'N':16,'K':16},'c',expected[(pid+1)*256:(pid+2)*256],label=f'full{pid}',
            descriptors=dict(desc,c_desc=desc2('c',48,16)),types={'c_desc':'tensordesc<fp32[16,16]>'},
            out_offset=(pid+1)*256,program_id=(pid,0,0),
            feeds={'c':('streamk_matmul/first','C') if pid==0 else ('streamk_matmul/full0','c')})
    add('streamk_matmul','full_tiles',{'a':('fp16',a),'bt':('fp16',bt),'c':('fp32',[0]*768)},
        {'M':48,'N':16,'K':16},'c',expected,label='early_return',preserve_output=True,
        descriptors=dict(desc,c_desc=desc2('c',48,16)),types={'c_desc':'tensordesc<fp32[16,16]>'},
        program_id=(2,0,0),feeds={'c':('streamk_matmul/full1','c')})
    return result
