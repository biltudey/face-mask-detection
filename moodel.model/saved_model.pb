งศ	
ิ
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype

Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
ม
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring จ
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.9.22v2.9.1-132-g18960c44ad38ำภ
~
Adam/dense_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_5/bias/v
w
'Adam/dense_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_5/bias/v*
_output_shapes
:*
dtype0

Adam/dense_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*&
shared_nameAdam/dense_5/kernel/v

)Adam/dense_5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_5/kernel/v*
_output_shapes

:2*
dtype0
~
Adam/dense_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*$
shared_nameAdam/dense_4/bias/v
w
'Adam/dense_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_4/bias/v*
_output_shapes
:2*
dtype0

Adam/dense_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ค2*&
shared_nameAdam/dense_4/kernel/v

)Adam/dense_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_4/kernel/v* 
_output_shapes
:
ค2*
dtype0

Adam/conv2d_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*%
shared_nameAdam/conv2d_5/bias/v
y
(Adam/conv2d_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_5/bias/v*
_output_shapes
:d*
dtype0

Adam/conv2d_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ศd*'
shared_nameAdam/conv2d_5/kernel/v

*Adam/conv2d_5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_5/kernel/v*'
_output_shapes
:ศd*
dtype0

Adam/conv2d_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ศ*%
shared_nameAdam/conv2d_4/bias/v
z
(Adam/conv2d_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_4/bias/v*
_output_shapes	
:ศ*
dtype0

Adam/conv2d_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ศ*'
shared_nameAdam/conv2d_4/kernel/v

*Adam/conv2d_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_4/kernel/v*'
_output_shapes
:ศ*
dtype0
~
Adam/dense_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_5/bias/m
w
'Adam/dense_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_5/bias/m*
_output_shapes
:*
dtype0

Adam/dense_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*&
shared_nameAdam/dense_5/kernel/m

)Adam/dense_5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_5/kernel/m*
_output_shapes

:2*
dtype0
~
Adam/dense_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*$
shared_nameAdam/dense_4/bias/m
w
'Adam/dense_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_4/bias/m*
_output_shapes
:2*
dtype0

Adam/dense_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ค2*&
shared_nameAdam/dense_4/kernel/m

)Adam/dense_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_4/kernel/m* 
_output_shapes
:
ค2*
dtype0

Adam/conv2d_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*%
shared_nameAdam/conv2d_5/bias/m
y
(Adam/conv2d_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_5/bias/m*
_output_shapes
:d*
dtype0

Adam/conv2d_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ศd*'
shared_nameAdam/conv2d_5/kernel/m

*Adam/conv2d_5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_5/kernel/m*'
_output_shapes
:ศd*
dtype0

Adam/conv2d_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ศ*%
shared_nameAdam/conv2d_4/bias/m
z
(Adam/conv2d_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_4/bias/m*
_output_shapes	
:ศ*
dtype0

Adam/conv2d_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ศ*'
shared_nameAdam/conv2d_4/kernel/m

*Adam/conv2d_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_4/kernel/m*'
_output_shapes
:ศ*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
p
dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_5/bias
i
 dense_5/bias/Read/ReadVariableOpReadVariableOpdense_5/bias*
_output_shapes
:*
dtype0
x
dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*
shared_namedense_5/kernel
q
"dense_5/kernel/Read/ReadVariableOpReadVariableOpdense_5/kernel*
_output_shapes

:2*
dtype0
p
dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*
shared_namedense_4/bias
i
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
_output_shapes
:2*
dtype0
z
dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ค2*
shared_namedense_4/kernel
s
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel* 
_output_shapes
:
ค2*
dtype0
r
conv2d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_nameconv2d_5/bias
k
!conv2d_5/bias/Read/ReadVariableOpReadVariableOpconv2d_5/bias*
_output_shapes
:d*
dtype0

conv2d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:ศd* 
shared_nameconv2d_5/kernel
|
#conv2d_5/kernel/Read/ReadVariableOpReadVariableOpconv2d_5/kernel*'
_output_shapes
:ศd*
dtype0
s
conv2d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ศ*
shared_nameconv2d_4/bias
l
!conv2d_4/bias/Read/ReadVariableOpReadVariableOpconv2d_4/bias*
_output_shapes	
:ศ*
dtype0

conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:ศ* 
shared_nameconv2d_4/kernel
|
#conv2d_4/kernel/Read/ReadVariableOpReadVariableOpconv2d_4/kernel*'
_output_shapes
:ศ*
dtype0

NoOpNoOp
ฟS
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*๚R
value๐RBํR BๆR
ถ
layer_with_weights-0
layer-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer-6
layer-7
	layer_with_weights-2
	layer-8

layer_with_weights-3

layer-9
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
ศ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias
 _jit_compiled_convolution_op*

	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses* 

#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses* 
ศ
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses

/kernel
0bias
 1_jit_compiled_convolution_op*

2	variables
3trainable_variables
4regularization_losses
5	keras_api
6__call__
*7&call_and_return_all_conditional_losses* 

8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses* 

>	variables
?trainable_variables
@regularization_losses
A	keras_api
B__call__
*C&call_and_return_all_conditional_losses* 
ฅ
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
H__call__
*I&call_and_return_all_conditional_losses
J_random_generator* 
ฆ
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
O__call__
*P&call_and_return_all_conditional_losses

Qkernel
Rbias*
ฆ
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W__call__
*X&call_and_return_all_conditional_losses

Ykernel
Zbias*
<
0
1
/2
03
Q4
R5
Y6
Z7*
<
0
1
/2
03
Q4
R5
Y6
Z7*
* 
ฐ
[non_trainable_variables

\layers
]metrics
^layer_regularization_losses
_layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
`trace_0
atrace_1
btrace_2
ctrace_3* 
6
dtrace_0
etrace_1
ftrace_2
gtrace_3* 
* 
ไ
hiter

ibeta_1

jbeta_2
	kdecay
llearning_ratemมmย/mร0mฤQmลRmฦYmวZmศvษvส/vห0vฬQvอRvฮYvฯZvะ*

mserving_default* 

0
1*

0
1*
* 

nnon_trainable_variables

olayers
pmetrics
qlayer_regularization_losses
rlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

strace_0* 

ttrace_0* 
_Y
VARIABLE_VALUEconv2d_4/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_4/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 

unon_trainable_variables

vlayers
wmetrics
xlayer_regularization_losses
ylayer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses* 

ztrace_0* 

{trace_0* 
* 
* 
* 

|non_trainable_variables

}layers
~metrics
layer_regularization_losses
layer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses* 

trace_0* 

trace_0* 

/0
01*

/0
01*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses*

trace_0* 

trace_0* 
_Y
VARIABLE_VALUEconv2d_5/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_5/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
2	variables
3trainable_variables
4regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses* 

trace_0* 

trace_0* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses* 

trace_0* 

trace_0* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
>	variables
?trainable_variables
@regularization_losses
B__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses* 

trace_0* 

trace_0* 
* 
* 
* 

non_trainable_variables
?layers
กmetrics
 ขlayer_regularization_losses
ฃlayer_metrics
D	variables
Etrainable_variables
Fregularization_losses
H__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses* 

คtrace_0
ฅtrace_1* 

ฆtrace_0
งtrace_1* 
* 

Q0
R1*

Q0
R1*
* 

จnon_trainable_variables
ฉlayers
ชmetrics
 ซlayer_regularization_losses
ฌlayer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses*

ญtrace_0* 

ฎtrace_0* 
^X
VARIABLE_VALUEdense_4/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_4/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

Y0
Z1*

Y0
Z1*
* 

ฏnon_trainable_variables
ฐlayers
ฑmetrics
 ฒlayer_regularization_losses
ณlayer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses*

ดtrace_0* 

ตtrace_0* 
^X
VARIABLE_VALUEdense_5/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_5/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
J
0
1
2
3
4
5
6
7
	8

9*

ถ0
ท1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
ธ	variables
น	keras_api

บtotal

ปcount*
M
ผ	variables
ฝ	keras_api

พtotal

ฟcount
ภ
_fn_kwargs*

บ0
ป1*

ธ	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

พ0
ฟ1*

ผ	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
|
VARIABLE_VALUEAdam/conv2d_4/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/conv2d_4/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/conv2d_5/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/conv2d_5/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAdam/dense_4/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_4/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAdam/dense_5/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_5/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/conv2d_4/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/conv2d_4/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/conv2d_5/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/conv2d_5/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAdam/dense_4/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_4/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAdam/dense_5/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_5/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

serving_default_conv2d_4_inputPlaceholder*/
_output_shapes
:?????????dd*
dtype0*$
shape:?????????dd
ฦ
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv2d_4_inputconv2d_4/kernelconv2d_4/biasconv2d_5/kernelconv2d_5/biasdense_4/kerneldense_4/biasdense_5/kerneldense_5/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference_signature_wrapper_6234
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#conv2d_4/kernel/Read/ReadVariableOp!conv2d_4/bias/Read/ReadVariableOp#conv2d_5/kernel/Read/ReadVariableOp!conv2d_5/bias/Read/ReadVariableOp"dense_4/kernel/Read/ReadVariableOp dense_4/bias/Read/ReadVariableOp"dense_5/kernel/Read/ReadVariableOp dense_5/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp*Adam/conv2d_4/kernel/m/Read/ReadVariableOp(Adam/conv2d_4/bias/m/Read/ReadVariableOp*Adam/conv2d_5/kernel/m/Read/ReadVariableOp(Adam/conv2d_5/bias/m/Read/ReadVariableOp)Adam/dense_4/kernel/m/Read/ReadVariableOp'Adam/dense_4/bias/m/Read/ReadVariableOp)Adam/dense_5/kernel/m/Read/ReadVariableOp'Adam/dense_5/bias/m/Read/ReadVariableOp*Adam/conv2d_4/kernel/v/Read/ReadVariableOp(Adam/conv2d_4/bias/v/Read/ReadVariableOp*Adam/conv2d_5/kernel/v/Read/ReadVariableOp(Adam/conv2d_5/bias/v/Read/ReadVariableOp)Adam/dense_4/kernel/v/Read/ReadVariableOp'Adam/dense_4/bias/v/Read/ReadVariableOp)Adam/dense_5/kernel/v/Read/ReadVariableOp'Adam/dense_5/bias/v/Read/ReadVariableOpConst*.
Tin'
%2#	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *&
f!R
__inference__traced_save_6635

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_4/kernelconv2d_4/biasconv2d_5/kernelconv2d_5/biasdense_4/kerneldense_4/biasdense_5/kerneldense_5/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_1count_1totalcountAdam/conv2d_4/kernel/mAdam/conv2d_4/bias/mAdam/conv2d_5/kernel/mAdam/conv2d_5/bias/mAdam/dense_4/kernel/mAdam/dense_4/bias/mAdam/dense_5/kernel/mAdam/dense_5/bias/mAdam/conv2d_4/kernel/vAdam/conv2d_4/bias/vAdam/conv2d_5/kernel/vAdam/conv2d_5/bias/vAdam/dense_4/kernel/vAdam/dense_4/bias/vAdam/dense_5/kernel/vAdam/dense_5/bias/v*-
Tin&
$2"*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *)
f$R"
 __inference__traced_restore_6744
ศ
G
+__inference_activation_4_layer_call_fn_6381

inputs
identityฝ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????bbศ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_activation_4_layer_call_and_return_conditional_losses_5875i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????bbศ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????bbศ:X T
0
_output_shapes
:?????????bbศ
 
_user_specified_nameinputs
๊	
ฯ
+__inference_sequential_2_layer_call_fn_6255

inputs"
unknown:ศ
	unknown_0:	ศ$
	unknown_1:ศd
	unknown_2:d
	unknown_3:
ค2
	unknown_4:2
	unknown_5:2
	unknown_6:
identityขStatefulPartitionedCallฌ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_sequential_2_layer_call_and_return_conditional_losses_5952o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????dd: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????dd
 
_user_specified_nameinputs
ญ

?
B__inference_conv2d_4_layer_call_and_return_conditional_losses_6376

inputs9
conv2d_readvariableop_resource:ศ.
biasadd_readvariableop_resource:	ศ
identityขBiasAdd/ReadVariableOpขConv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:ศ*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????bbศ*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ศ*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????bbศh
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:?????????bbศw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????dd: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????dd
 
_user_specified_nameinputs

e
I__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_5832

inputs
identityข
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
)

F__inference_sequential_2_layer_call_and_return_conditional_losses_6105

inputs(
conv2d_4_6078:ศ
conv2d_4_6080:	ศ(
conv2d_5_6085:ศd
conv2d_5_6087:d 
dense_4_6094:
ค2
dense_4_6096:2
dense_5_6099:2
dense_5_6101:
identityข conv2d_4/StatefulPartitionedCallข conv2d_5/StatefulPartitionedCallขdense_4/StatefulPartitionedCallขdense_5/StatefulPartitionedCallข!dropout_2/StatefulPartitionedCall๖
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_4_6078conv2d_4_6080*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????bbศ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_4_layer_call_and_return_conditional_losses_5864ํ
activation_4/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????bbศ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_activation_4_layer_call_and_return_conditional_losses_5875๏
max_pooling2d_4/PartitionedCallPartitionedCall%activation_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????11ศ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_5832
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_4/PartitionedCall:output:0conv2d_5_6085conv2d_5_6087*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????//d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_5_layer_call_and_return_conditional_losses_5888์
activation_5/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????//d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_activation_5_layer_call_and_return_conditional_losses_5899๎
max_pooling2d_5/PartitionedCallPartitionedCall%activation_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_5844฿
flatten_2/PartitionedCallPartitionedCall(max_pooling2d_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:?????????ค* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_flatten_2_layer_call_and_return_conditional_losses_5908้
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:?????????ค* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dropout_2_layer_call_and_return_conditional_losses_6011
dense_4/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0dense_4_6094dense_4_6096*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_4_layer_call_and_return_conditional_losses_5928
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_6099dense_5_6101*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_5_layer_call_and_return_conditional_losses_5945w
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????๔
NoOpNoOp!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????dd: : : : : : : : 2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall:W S
/
_output_shapes
:?????????dd
 
_user_specified_nameinputs


ื
+__inference_sequential_2_layer_call_fn_6145
conv2d_4_input"
unknown:ศ
	unknown_0:	ศ$
	unknown_1:ศd
	unknown_2:d
	unknown_3:
ค2
	unknown_4:2
	unknown_5:2
	unknown_6:
identityขStatefulPartitionedCallด
StatefulPartitionedCallStatefulPartitionedCallconv2d_4_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_sequential_2_layer_call_and_return_conditional_losses_6105o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????dd: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:?????????dd
(
_user_specified_nameconv2d_4_input
ฤ
G
+__inference_activation_5_layer_call_fn_6420

inputs
identityผ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????//d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_activation_5_layer_call_and_return_conditional_losses_5899h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????//d"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????//d:W S
/
_output_shapes
:?????????//d
 
_user_specified_nameinputs
?'

F__inference_sequential_2_layer_call_and_return_conditional_losses_6175
conv2d_4_input(
conv2d_4_6148:ศ
conv2d_4_6150:	ศ(
conv2d_5_6155:ศd
conv2d_5_6157:d 
dense_4_6164:
ค2
dense_4_6166:2
dense_5_6169:2
dense_5_6171:
identityข conv2d_4/StatefulPartitionedCallข conv2d_5/StatefulPartitionedCallขdense_4/StatefulPartitionedCallขdense_5/StatefulPartitionedCall?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCallconv2d_4_inputconv2d_4_6148conv2d_4_6150*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????bbศ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_4_layer_call_and_return_conditional_losses_5864ํ
activation_4/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????bbศ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_activation_4_layer_call_and_return_conditional_losses_5875๏
max_pooling2d_4/PartitionedCallPartitionedCall%activation_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????11ศ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_5832
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_4/PartitionedCall:output:0conv2d_5_6155conv2d_5_6157*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????//d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_5_layer_call_and_return_conditional_losses_5888์
activation_5/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????//d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_activation_5_layer_call_and_return_conditional_losses_5899๎
max_pooling2d_5/PartitionedCallPartitionedCall%activation_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_5844฿
flatten_2/PartitionedCallPartitionedCall(max_pooling2d_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:?????????ค* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_flatten_2_layer_call_and_return_conditional_losses_5908ู
dropout_2/PartitionedCallPartitionedCall"flatten_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:?????????ค* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dropout_2_layer_call_and_return_conditional_losses_5915
dense_4/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0dense_4_6164dense_4_6166*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_4_layer_call_and_return_conditional_losses_5928
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_6169dense_5_6171*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_5_layer_call_and_return_conditional_losses_5945w
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????ะ
NoOpNoOp!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????dd: : : : : : : : 2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:_ [
/
_output_shapes
:?????????dd
(
_user_specified_nameconv2d_4_input
ฒ
D
(__inference_flatten_2_layer_call_fn_6440

inputs
identityณ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:?????????ค* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_flatten_2_layer_call_and_return_conditional_losses_5908b
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:?????????ค"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????d:W S
/
_output_shapes
:?????????d
 
_user_specified_nameinputs


๒
A__inference_dense_5_layer_call_and_return_conditional_losses_5945

inputs0
matmul_readvariableop_resource:2-
biasadd_readvariableop_resource:
identityขBiasAdd/ReadVariableOpขMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????2: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
ท
J
.__inference_max_pooling2d_4_layer_call_fn_6391

inputs
identityฺ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_5832
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
ว
_
C__inference_flatten_2_layer_call_and_return_conditional_losses_5908

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"????คฮ  ^
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:?????????คZ
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:?????????ค"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????d:W S
/
_output_shapes
:?????????d
 
_user_specified_nameinputs


ื
+__inference_sequential_2_layer_call_fn_5971
conv2d_4_input"
unknown:ศ
	unknown_0:	ศ$
	unknown_1:ศd
	unknown_2:d
	unknown_3:
ค2
	unknown_4:2
	unknown_5:2
	unknown_6:
identityขStatefulPartitionedCallด
StatefulPartitionedCallStatefulPartitionedCallconv2d_4_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_sequential_2_layer_call_and_return_conditional_losses_5952o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????dd: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:?????????dd
(
_user_specified_nameconv2d_4_input
๊
b
F__inference_activation_5_layer_call_and_return_conditional_losses_6425

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:?????????//db
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????//d"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????//d:W S
/
_output_shapes
:?????????//d
 
_user_specified_nameinputs
ว
_
C__inference_flatten_2_layer_call_and_return_conditional_losses_6446

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"????คฮ  ^
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:?????????คZ
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:?????????ค"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????d:W S
/
_output_shapes
:?????????d
 
_user_specified_nameinputs
๘
a
(__inference_dropout_2_layer_call_fn_6456

inputs
identityขStatefulPartitionedCallร
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:?????????ค* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dropout_2_layer_call_and_return_conditional_losses_6011q
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*)
_output_shapes
:?????????ค`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????ค22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:?????????ค
 
_user_specified_nameinputs
ล

&__inference_dense_4_layer_call_fn_6482

inputs
unknown:
ค2
	unknown_0:2
identityขStatefulPartitionedCallู
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_4_layer_call_and_return_conditional_losses_5928o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:?????????ค: : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:?????????ค
 
_user_specified_nameinputs
ๆ'
๘
F__inference_sequential_2_layer_call_and_return_conditional_losses_5952

inputs(
conv2d_4_5865:ศ
conv2d_4_5867:	ศ(
conv2d_5_5889:ศd
conv2d_5_5891:d 
dense_4_5929:
ค2
dense_4_5931:2
dense_5_5946:2
dense_5_5948:
identityข conv2d_4/StatefulPartitionedCallข conv2d_5/StatefulPartitionedCallขdense_4/StatefulPartitionedCallขdense_5/StatefulPartitionedCall๖
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_4_5865conv2d_4_5867*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????bbศ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_4_layer_call_and_return_conditional_losses_5864ํ
activation_4/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????bbศ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_activation_4_layer_call_and_return_conditional_losses_5875๏
max_pooling2d_4/PartitionedCallPartitionedCall%activation_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????11ศ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_5832
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_4/PartitionedCall:output:0conv2d_5_5889conv2d_5_5891*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????//d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_5_layer_call_and_return_conditional_losses_5888์
activation_5/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????//d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_activation_5_layer_call_and_return_conditional_losses_5899๎
max_pooling2d_5/PartitionedCallPartitionedCall%activation_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_5844฿
flatten_2/PartitionedCallPartitionedCall(max_pooling2d_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:?????????ค* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_flatten_2_layer_call_and_return_conditional_losses_5908ู
dropout_2/PartitionedCallPartitionedCall"flatten_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:?????????ค* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dropout_2_layer_call_and_return_conditional_losses_5915
dense_4/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0dense_4_5929dense_4_5931*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_4_layer_call_and_return_conditional_losses_5928
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_5946dense_5_5948*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_5_layer_call_and_return_conditional_losses_5945w
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????ะ
NoOpNoOp!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????dd: : : : : : : : 2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:W S
/
_output_shapes
:?????????dd
 
_user_specified_nameinputs
ง+
ส
F__inference_sequential_2_layer_call_and_return_conditional_losses_6313

inputsB
'conv2d_4_conv2d_readvariableop_resource:ศ7
(conv2d_4_biasadd_readvariableop_resource:	ศB
'conv2d_5_conv2d_readvariableop_resource:ศd6
(conv2d_5_biasadd_readvariableop_resource:d:
&dense_4_matmul_readvariableop_resource:
ค25
'dense_4_biasadd_readvariableop_resource:28
&dense_5_matmul_readvariableop_resource:25
'dense_5_biasadd_readvariableop_resource:
identityขconv2d_4/BiasAdd/ReadVariableOpขconv2d_4/Conv2D/ReadVariableOpขconv2d_5/BiasAdd/ReadVariableOpขconv2d_5/Conv2D/ReadVariableOpขdense_4/BiasAdd/ReadVariableOpขdense_4/MatMul/ReadVariableOpขdense_5/BiasAdd/ReadVariableOpขdense_5/MatMul/ReadVariableOp
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*'
_output_shapes
:ศ*
dtype0ญ
conv2d_4/Conv2DConv2Dinputs&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????bbศ*
paddingVALID*
strides

conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:ศ*
dtype0
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????bbศo
activation_4/ReluReluconv2d_4/BiasAdd:output:0*
T0*0
_output_shapes
:?????????bbศฑ
max_pooling2d_4/MaxPoolMaxPoolactivation_4/Relu:activations:0*0
_output_shapes
:?????????11ศ*
ksize
*
paddingVALID*
strides

conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*'
_output_shapes
:ศd*
dtype0ฦ
conv2d_5/Conv2DConv2D max_pooling2d_4/MaxPool:output:0&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????//d*
paddingVALID*
strides

conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????//dn
activation_5/ReluReluconv2d_5/BiasAdd:output:0*
T0*/
_output_shapes
:?????????//dฐ
max_pooling2d_5/MaxPoolMaxPoolactivation_5/Relu:activations:0*/
_output_shapes
:?????????d*
ksize
*
paddingVALID*
strides
`
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"????คฮ  
flatten_2/ReshapeReshape max_pooling2d_5/MaxPool:output:0flatten_2/Const:output:0*
T0*)
_output_shapes
:?????????คn
dropout_2/IdentityIdentityflatten_2/Reshape:output:0*
T0*)
_output_shapes
:?????????ค
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource* 
_output_shapes
:
ค2*
dtype0
dense_4/MatMulMatMuldropout_2/Identity:output:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2`
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0
dense_5/MatMulMatMuldense_4/Relu:activations:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f
dense_5/SigmoidSigmoiddense_5/BiasAdd:output:0*
T0*'
_output_shapes
:?????????b
IdentityIdentitydense_5/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????ฮ
NoOpNoOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????dd: : : : : : : : 2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2B
conv2d_5/BiasAdd/ReadVariableOpconv2d_5/BiasAdd/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????dd
 
_user_specified_nameinputs
ญ

?
B__inference_conv2d_4_layer_call_and_return_conditional_losses_5864

inputs9
conv2d_readvariableop_resource:ศ.
biasadd_readvariableop_resource:	ศ
identityขBiasAdd/ReadVariableOpขConv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:ศ*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????bbศ*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ศ*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????bbศh
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:?????????bbศw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????dd: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????dd
 
_user_specified_nameinputs
์

'__inference_conv2d_5_layer_call_fn_6405

inputs"
unknown:ศd
	unknown_0:d
identityขStatefulPartitionedCallโ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????//d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_5_layer_call_and_return_conditional_losses_5888w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????//d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????11ศ: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????11ศ
 
_user_specified_nameinputs
?
a
C__inference_dropout_2_layer_call_and_return_conditional_losses_5915

inputs

identity_1P
IdentityIdentityinputs*
T0*)
_output_shapes
:?????????ค]

Identity_1IdentityIdentity:output:0*
T0*)
_output_shapes
:?????????ค"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????ค:Q M
)
_output_shapes
:?????????ค
 
_user_specified_nameinputs
๎
b
F__inference_activation_4_layer_call_and_return_conditional_losses_5875

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:?????????bbศc
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????bbศ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????bbศ:X T
0
_output_shapes
:?????????bbศ
 
_user_specified_nameinputs
า	
ฮ
"__inference_signature_wrapper_6234
conv2d_4_input"
unknown:ศ
	unknown_0:	ศ$
	unknown_1:ศd
	unknown_2:d
	unknown_3:
ค2
	unknown_4:2
	unknown_5:2
	unknown_6:
identityขStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallconv2d_4_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *(
f#R!
__inference__wrapped_model_5823o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????dd: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:?????????dd
(
_user_specified_nameconv2d_4_input
๊
b
F__inference_activation_5_layer_call_and_return_conditional_losses_5899

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:?????????//db
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????//d"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????//d:W S
/
_output_shapes
:?????????//d
 
_user_specified_nameinputs


๒
A__inference_dense_5_layer_call_and_return_conditional_losses_6513

inputs0
matmul_readvariableop_resource:2-
biasadd_readvariableop_resource:
identityขBiasAdd/ReadVariableOpขMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????2: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
ฆ)
ค
F__inference_sequential_2_layer_call_and_return_conditional_losses_6205
conv2d_4_input(
conv2d_4_6178:ศ
conv2d_4_6180:	ศ(
conv2d_5_6185:ศd
conv2d_5_6187:d 
dense_4_6194:
ค2
dense_4_6196:2
dense_5_6199:2
dense_5_6201:
identityข conv2d_4/StatefulPartitionedCallข conv2d_5/StatefulPartitionedCallขdense_4/StatefulPartitionedCallขdense_5/StatefulPartitionedCallข!dropout_2/StatefulPartitionedCall?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCallconv2d_4_inputconv2d_4_6178conv2d_4_6180*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????bbศ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_4_layer_call_and_return_conditional_losses_5864ํ
activation_4/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????bbศ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_activation_4_layer_call_and_return_conditional_losses_5875๏
max_pooling2d_4/PartitionedCallPartitionedCall%activation_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????11ศ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_5832
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_4/PartitionedCall:output:0conv2d_5_6185conv2d_5_6187*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????//d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_5_layer_call_and_return_conditional_losses_5888์
activation_5/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????//d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_activation_5_layer_call_and_return_conditional_losses_5899๎
max_pooling2d_5/PartitionedCallPartitionedCall%activation_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_5844฿
flatten_2/PartitionedCallPartitionedCall(max_pooling2d_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:?????????ค* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_flatten_2_layer_call_and_return_conditional_losses_5908้
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:?????????ค* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dropout_2_layer_call_and_return_conditional_losses_6011
dense_4/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0dense_4_6194dense_4_6196*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_4_layer_call_and_return_conditional_losses_5928
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_6199dense_5_6201*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_5_layer_call_and_return_conditional_losses_5945w
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????๔
NoOpNoOp!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????dd: : : : : : : : 2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall:_ [
/
_output_shapes
:?????????dd
(
_user_specified_nameconv2d_4_input
ฟ

&__inference_dense_5_layer_call_fn_6502

inputs
unknown:2
	unknown_0:
identityขStatefulPartitionedCallู
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_5_layer_call_and_return_conditional_losses_5945o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????2: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs

e
I__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_6435

inputs
identityข
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
ช

?
B__inference_conv2d_5_layer_call_and_return_conditional_losses_5888

inputs9
conv2d_readvariableop_resource:ศd-
biasadd_readvariableop_resource:d
identityขBiasAdd/ReadVariableOpขConv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:ศd*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????//d*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????//dg
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????//dw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????11ศ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????11ศ
 
_user_specified_nameinputs
ฆ
D
(__inference_dropout_2_layer_call_fn_6451

inputs
identityณ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:?????????ค* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dropout_2_layer_call_and_return_conditional_losses_5915b
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:?????????ค"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????ค:Q M
)
_output_shapes
:?????????ค
 
_user_specified_nameinputs
?

๔
A__inference_dense_4_layer_call_and_return_conditional_losses_6493

inputs2
matmul_readvariableop_resource:
ค2-
biasadd_readvariableop_resource:2
identityขBiasAdd/ReadVariableOpขMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ค2*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????2w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:?????????ค: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:?????????ค
 
_user_specified_nameinputs

e
I__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_6396

inputs
identityข
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
ๆF
ฒ
__inference__traced_save_6635
file_prefix.
*savev2_conv2d_4_kernel_read_readvariableop,
(savev2_conv2d_4_bias_read_readvariableop.
*savev2_conv2d_5_kernel_read_readvariableop,
(savev2_conv2d_5_bias_read_readvariableop-
)savev2_dense_4_kernel_read_readvariableop+
'savev2_dense_4_bias_read_readvariableop-
)savev2_dense_5_kernel_read_readvariableop+
'savev2_dense_5_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop5
1savev2_adam_conv2d_4_kernel_m_read_readvariableop3
/savev2_adam_conv2d_4_bias_m_read_readvariableop5
1savev2_adam_conv2d_5_kernel_m_read_readvariableop3
/savev2_adam_conv2d_5_bias_m_read_readvariableop4
0savev2_adam_dense_4_kernel_m_read_readvariableop2
.savev2_adam_dense_4_bias_m_read_readvariableop4
0savev2_adam_dense_5_kernel_m_read_readvariableop2
.savev2_adam_dense_5_bias_m_read_readvariableop5
1savev2_adam_conv2d_4_kernel_v_read_readvariableop3
/savev2_adam_conv2d_4_bias_v_read_readvariableop5
1savev2_adam_conv2d_5_kernel_v_read_readvariableop3
/savev2_adam_conv2d_5_bias_v_read_readvariableop4
0savev2_adam_dense_4_kernel_v_read_readvariableop2
.savev2_adam_dense_4_bias_v_read_readvariableop4
0savev2_adam_dense_5_kernel_v_read_readvariableop2
.savev2_adam_dense_5_bias_v_read_readvariableop
savev2_const

identity_1ขMergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ฏ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*ุ
valueฮBห"B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHฑ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_conv2d_4_kernel_read_readvariableop(savev2_conv2d_4_bias_read_readvariableop*savev2_conv2d_5_kernel_read_readvariableop(savev2_conv2d_5_bias_read_readvariableop)savev2_dense_4_kernel_read_readvariableop'savev2_dense_4_bias_read_readvariableop)savev2_dense_5_kernel_read_readvariableop'savev2_dense_5_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop1savev2_adam_conv2d_4_kernel_m_read_readvariableop/savev2_adam_conv2d_4_bias_m_read_readvariableop1savev2_adam_conv2d_5_kernel_m_read_readvariableop/savev2_adam_conv2d_5_bias_m_read_readvariableop0savev2_adam_dense_4_kernel_m_read_readvariableop.savev2_adam_dense_4_bias_m_read_readvariableop0savev2_adam_dense_5_kernel_m_read_readvariableop.savev2_adam_dense_5_bias_m_read_readvariableop1savev2_adam_conv2d_4_kernel_v_read_readvariableop/savev2_adam_conv2d_4_bias_v_read_readvariableop1savev2_adam_conv2d_5_kernel_v_read_readvariableop/savev2_adam_conv2d_5_bias_v_read_readvariableop0savev2_adam_dense_4_kernel_v_read_readvariableop.savev2_adam_dense_4_bias_v_read_readvariableop0savev2_adam_dense_5_kernel_v_read_readvariableop.savev2_adam_dense_5_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *0
dtypes&
$2"	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*ช
_input_shapes
: :ศ:ศ:ศd:d:
ค2:2:2:: : : : : : : : : :ศ:ศ:ศd:d:
ค2:2:2::ศ:ศ:ศd:d:
ค2:2:2:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:-)
'
_output_shapes
:ศ:!

_output_shapes	
:ศ:-)
'
_output_shapes
:ศd: 

_output_shapes
:d:&"
 
_output_shapes
:
ค2: 

_output_shapes
:2:$ 

_output_shapes

:2: 

_output_shapes
::	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ศ:!

_output_shapes	
:ศ:-)
'
_output_shapes
:ศd: 

_output_shapes
:d:&"
 
_output_shapes
:
ค2: 

_output_shapes
:2:$ 

_output_shapes

:2: 

_output_shapes
::-)
'
_output_shapes
:ศ:!

_output_shapes	
:ศ:-)
'
_output_shapes
:ศd: 

_output_shapes
:d:&"
 
_output_shapes
:
ค2: 

_output_shapes
:2:$  

_output_shapes

:2: !

_output_shapes
::"

_output_shapes
: 

e
I__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_5844

inputs
identityข
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?

๔
A__inference_dense_4_layer_call_and_return_conditional_losses_5928

inputs2
matmul_readvariableop_resource:
ค2-
biasadd_readvariableop_resource:2
identityขBiasAdd/ReadVariableOpขMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ค2*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????2w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:?????????ค: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:?????????ค
 
_user_specified_nameinputs
๒2
ส
F__inference_sequential_2_layer_call_and_return_conditional_losses_6357

inputsB
'conv2d_4_conv2d_readvariableop_resource:ศ7
(conv2d_4_biasadd_readvariableop_resource:	ศB
'conv2d_5_conv2d_readvariableop_resource:ศd6
(conv2d_5_biasadd_readvariableop_resource:d:
&dense_4_matmul_readvariableop_resource:
ค25
'dense_4_biasadd_readvariableop_resource:28
&dense_5_matmul_readvariableop_resource:25
'dense_5_biasadd_readvariableop_resource:
identityขconv2d_4/BiasAdd/ReadVariableOpขconv2d_4/Conv2D/ReadVariableOpขconv2d_5/BiasAdd/ReadVariableOpขconv2d_5/Conv2D/ReadVariableOpขdense_4/BiasAdd/ReadVariableOpขdense_4/MatMul/ReadVariableOpขdense_5/BiasAdd/ReadVariableOpขdense_5/MatMul/ReadVariableOp
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*'
_output_shapes
:ศ*
dtype0ญ
conv2d_4/Conv2DConv2Dinputs&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????bbศ*
paddingVALID*
strides

conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:ศ*
dtype0
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????bbศo
activation_4/ReluReluconv2d_4/BiasAdd:output:0*
T0*0
_output_shapes
:?????????bbศฑ
max_pooling2d_4/MaxPoolMaxPoolactivation_4/Relu:activations:0*0
_output_shapes
:?????????11ศ*
ksize
*
paddingVALID*
strides

conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*'
_output_shapes
:ศd*
dtype0ฦ
conv2d_5/Conv2DConv2D max_pooling2d_4/MaxPool:output:0&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????//d*
paddingVALID*
strides

conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????//dn
activation_5/ReluReluconv2d_5/BiasAdd:output:0*
T0*/
_output_shapes
:?????????//dฐ
max_pooling2d_5/MaxPoolMaxPoolactivation_5/Relu:activations:0*/
_output_shapes
:?????????d*
ksize
*
paddingVALID*
strides
`
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"????คฮ  
flatten_2/ReshapeReshape max_pooling2d_5/MaxPool:output:0flatten_2/Const:output:0*
T0*)
_output_shapes
:?????????ค\
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @
dropout_2/dropout/MulMulflatten_2/Reshape:output:0 dropout_2/dropout/Const:output:0*
T0*)
_output_shapes
:?????????คa
dropout_2/dropout/ShapeShapeflatten_2/Reshape:output:0*
T0*
_output_shapes
:ข
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*)
_output_shapes
:?????????ค*
dtype0e
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?ฦ
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*)
_output_shapes
:?????????ค
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*)
_output_shapes
:?????????ค
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*)
_output_shapes
:?????????ค
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource* 
_output_shapes
:
ค2*
dtype0
dense_4/MatMulMatMuldropout_2/dropout/Mul_1:z:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2`
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0
dense_5/MatMulMatMuldense_4/Relu:activations:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f
dense_5/SigmoidSigmoiddense_5/BiasAdd:output:0*
T0*'
_output_shapes
:?????????b
IdentityIdentitydense_5/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????ฮ
NoOpNoOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????dd: : : : : : : : 2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2B
conv2d_5/BiasAdd/ReadVariableOpconv2d_5/BiasAdd/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????dd
 
_user_specified_nameinputs
๊	
ฯ
+__inference_sequential_2_layer_call_fn_6276

inputs"
unknown:ศ
	unknown_0:	ศ$
	unknown_1:ศd
	unknown_2:d
	unknown_3:
ค2
	unknown_4:2
	unknown_5:2
	unknown_6:
identityขStatefulPartitionedCallฌ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_sequential_2_layer_call_and_return_conditional_losses_6105o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????dd: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????dd
 
_user_specified_nameinputs
ํ

'__inference_conv2d_4_layer_call_fn_6366

inputs"
unknown:ศ
	unknown_0:	ศ
identityขStatefulPartitionedCallใ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????bbศ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_4_layer_call_and_return_conditional_losses_5864x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????bbศ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????dd: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????dd
 
_user_specified_nameinputs
5
๛
__inference__wrapped_model_5823
conv2d_4_inputO
4sequential_2_conv2d_4_conv2d_readvariableop_resource:ศD
5sequential_2_conv2d_4_biasadd_readvariableop_resource:	ศO
4sequential_2_conv2d_5_conv2d_readvariableop_resource:ศdC
5sequential_2_conv2d_5_biasadd_readvariableop_resource:dG
3sequential_2_dense_4_matmul_readvariableop_resource:
ค2B
4sequential_2_dense_4_biasadd_readvariableop_resource:2E
3sequential_2_dense_5_matmul_readvariableop_resource:2B
4sequential_2_dense_5_biasadd_readvariableop_resource:
identityข,sequential_2/conv2d_4/BiasAdd/ReadVariableOpข+sequential_2/conv2d_4/Conv2D/ReadVariableOpข,sequential_2/conv2d_5/BiasAdd/ReadVariableOpข+sequential_2/conv2d_5/Conv2D/ReadVariableOpข+sequential_2/dense_4/BiasAdd/ReadVariableOpข*sequential_2/dense_4/MatMul/ReadVariableOpข+sequential_2/dense_5/BiasAdd/ReadVariableOpข*sequential_2/dense_5/MatMul/ReadVariableOpฉ
+sequential_2/conv2d_4/Conv2D/ReadVariableOpReadVariableOp4sequential_2_conv2d_4_conv2d_readvariableop_resource*'
_output_shapes
:ศ*
dtype0ฯ
sequential_2/conv2d_4/Conv2DConv2Dconv2d_4_input3sequential_2/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????bbศ*
paddingVALID*
strides

,sequential_2/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:ศ*
dtype0ภ
sequential_2/conv2d_4/BiasAddBiasAdd%sequential_2/conv2d_4/Conv2D:output:04sequential_2/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????bbศ
sequential_2/activation_4/ReluRelu&sequential_2/conv2d_4/BiasAdd:output:0*
T0*0
_output_shapes
:?????????bbศห
$sequential_2/max_pooling2d_4/MaxPoolMaxPool,sequential_2/activation_4/Relu:activations:0*0
_output_shapes
:?????????11ศ*
ksize
*
paddingVALID*
strides
ฉ
+sequential_2/conv2d_5/Conv2D/ReadVariableOpReadVariableOp4sequential_2_conv2d_5_conv2d_readvariableop_resource*'
_output_shapes
:ศd*
dtype0ํ
sequential_2/conv2d_5/Conv2DConv2D-sequential_2/max_pooling2d_4/MaxPool:output:03sequential_2/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????//d*
paddingVALID*
strides

,sequential_2/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0ฟ
sequential_2/conv2d_5/BiasAddBiasAdd%sequential_2/conv2d_5/Conv2D:output:04sequential_2/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????//d
sequential_2/activation_5/ReluRelu&sequential_2/conv2d_5/BiasAdd:output:0*
T0*/
_output_shapes
:?????????//dส
$sequential_2/max_pooling2d_5/MaxPoolMaxPool,sequential_2/activation_5/Relu:activations:0*/
_output_shapes
:?????????d*
ksize
*
paddingVALID*
strides
m
sequential_2/flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"????คฮ  ณ
sequential_2/flatten_2/ReshapeReshape-sequential_2/max_pooling2d_5/MaxPool:output:0%sequential_2/flatten_2/Const:output:0*
T0*)
_output_shapes
:?????????ค
sequential_2/dropout_2/IdentityIdentity'sequential_2/flatten_2/Reshape:output:0*
T0*)
_output_shapes
:?????????ค?
*sequential_2/dense_4/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_4_matmul_readvariableop_resource* 
_output_shapes
:
ค2*
dtype0ต
sequential_2/dense_4/MatMulMatMul(sequential_2/dropout_2/Identity:output:02sequential_2/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
+sequential_2/dense_4/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_4_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0ต
sequential_2/dense_4/BiasAddBiasAdd%sequential_2/dense_4/MatMul:product:03sequential_2/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2z
sequential_2/dense_4/ReluRelu%sequential_2/dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
*sequential_2/dense_5/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_5_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0ด
sequential_2/dense_5/MatMulMatMul'sequential_2/dense_4/Relu:activations:02sequential_2/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
+sequential_2/dense_5/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ต
sequential_2/dense_5/BiasAddBiasAdd%sequential_2/dense_5/MatMul:product:03sequential_2/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
sequential_2/dense_5/SigmoidSigmoid%sequential_2/dense_5/BiasAdd:output:0*
T0*'
_output_shapes
:?????????o
IdentityIdentity sequential_2/dense_5/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????ถ
NoOpNoOp-^sequential_2/conv2d_4/BiasAdd/ReadVariableOp,^sequential_2/conv2d_4/Conv2D/ReadVariableOp-^sequential_2/conv2d_5/BiasAdd/ReadVariableOp,^sequential_2/conv2d_5/Conv2D/ReadVariableOp,^sequential_2/dense_4/BiasAdd/ReadVariableOp+^sequential_2/dense_4/MatMul/ReadVariableOp,^sequential_2/dense_5/BiasAdd/ReadVariableOp+^sequential_2/dense_5/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????dd: : : : : : : : 2\
,sequential_2/conv2d_4/BiasAdd/ReadVariableOp,sequential_2/conv2d_4/BiasAdd/ReadVariableOp2Z
+sequential_2/conv2d_4/Conv2D/ReadVariableOp+sequential_2/conv2d_4/Conv2D/ReadVariableOp2\
,sequential_2/conv2d_5/BiasAdd/ReadVariableOp,sequential_2/conv2d_5/BiasAdd/ReadVariableOp2Z
+sequential_2/conv2d_5/Conv2D/ReadVariableOp+sequential_2/conv2d_5/Conv2D/ReadVariableOp2Z
+sequential_2/dense_4/BiasAdd/ReadVariableOp+sequential_2/dense_4/BiasAdd/ReadVariableOp2X
*sequential_2/dense_4/MatMul/ReadVariableOp*sequential_2/dense_4/MatMul/ReadVariableOp2Z
+sequential_2/dense_5/BiasAdd/ReadVariableOp+sequential_2/dense_5/BiasAdd/ReadVariableOp2X
*sequential_2/dense_5/MatMul/ReadVariableOp*sequential_2/dense_5/MatMul/ReadVariableOp:_ [
/
_output_shapes
:?????????dd
(
_user_specified_nameconv2d_4_input
ท
J
.__inference_max_pooling2d_5_layer_call_fn_6430

inputs
identityฺ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_5844
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
a
C__inference_dropout_2_layer_call_and_return_conditional_losses_6461

inputs

identity_1P
IdentityIdentityinputs*
T0*)
_output_shapes
:?????????ค]

Identity_1IdentityIdentity:output:0*
T0*)
_output_shapes
:?????????ค"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????ค:Q M
)
_output_shapes
:?????????ค
 
_user_specified_nameinputs


b
C__inference_dropout_2_layer_call_and_return_conditional_losses_6473

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @f
dropout/MulMulinputsdropout/Const:output:0*
T0*)
_output_shapes
:?????????คC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*)
_output_shapes
:?????????ค*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?จ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*)
_output_shapes
:?????????คq
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*)
_output_shapes
:?????????คk
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*)
_output_shapes
:?????????ค[
IdentityIdentitydropout/Mul_1:z:0*
T0*)
_output_shapes
:?????????ค"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????ค:Q M
)
_output_shapes
:?????????ค
 
_user_specified_nameinputs
ช

?
B__inference_conv2d_5_layer_call_and_return_conditional_losses_6415

inputs9
conv2d_readvariableop_resource:ศd-
biasadd_readvariableop_resource:d
identityขBiasAdd/ReadVariableOpขConv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:ศd*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????//d*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????//dg
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????//dw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????11ศ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????11ศ
 
_user_specified_nameinputs


b
C__inference_dropout_2_layer_call_and_return_conditional_losses_6011

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @f
dropout/MulMulinputsdropout/Const:output:0*
T0*)
_output_shapes
:?????????คC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*)
_output_shapes
:?????????ค*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?จ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*)
_output_shapes
:?????????คq
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*)
_output_shapes
:?????????คk
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*)
_output_shapes
:?????????ค[
IdentityIdentitydropout/Mul_1:z:0*
T0*)
_output_shapes
:?????????ค"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????ค:Q M
)
_output_shapes
:?????????ค
 
_user_specified_nameinputs
ฎ
๛
 __inference__traced_restore_6744
file_prefix;
 assignvariableop_conv2d_4_kernel:ศ/
 assignvariableop_1_conv2d_4_bias:	ศ=
"assignvariableop_2_conv2d_5_kernel:ศd.
 assignvariableop_3_conv2d_5_bias:d5
!assignvariableop_4_dense_4_kernel:
ค2-
assignvariableop_5_dense_4_bias:23
!assignvariableop_6_dense_5_kernel:2-
assignvariableop_7_dense_5_bias:&
assignvariableop_8_adam_iter:	 (
assignvariableop_9_adam_beta_1: )
assignvariableop_10_adam_beta_2: (
assignvariableop_11_adam_decay: 0
&assignvariableop_12_adam_learning_rate: %
assignvariableop_13_total_1: %
assignvariableop_14_count_1: #
assignvariableop_15_total: #
assignvariableop_16_count: E
*assignvariableop_17_adam_conv2d_4_kernel_m:ศ7
(assignvariableop_18_adam_conv2d_4_bias_m:	ศE
*assignvariableop_19_adam_conv2d_5_kernel_m:ศd6
(assignvariableop_20_adam_conv2d_5_bias_m:d=
)assignvariableop_21_adam_dense_4_kernel_m:
ค25
'assignvariableop_22_adam_dense_4_bias_m:2;
)assignvariableop_23_adam_dense_5_kernel_m:25
'assignvariableop_24_adam_dense_5_bias_m:E
*assignvariableop_25_adam_conv2d_4_kernel_v:ศ7
(assignvariableop_26_adam_conv2d_4_bias_v:	ศE
*assignvariableop_27_adam_conv2d_5_kernel_v:ศd6
(assignvariableop_28_adam_conv2d_5_bias_v:d=
)assignvariableop_29_adam_dense_4_kernel_v:
ค25
'assignvariableop_30_adam_dense_4_bias_v:2;
)assignvariableop_31_adam_dense_5_kernel_v:25
'assignvariableop_32_adam_dense_5_bias_v:
identity_34ขAssignVariableOpขAssignVariableOp_1ขAssignVariableOp_10ขAssignVariableOp_11ขAssignVariableOp_12ขAssignVariableOp_13ขAssignVariableOp_14ขAssignVariableOp_15ขAssignVariableOp_16ขAssignVariableOp_17ขAssignVariableOp_18ขAssignVariableOp_19ขAssignVariableOp_2ขAssignVariableOp_20ขAssignVariableOp_21ขAssignVariableOp_22ขAssignVariableOp_23ขAssignVariableOp_24ขAssignVariableOp_25ขAssignVariableOp_26ขAssignVariableOp_27ขAssignVariableOp_28ขAssignVariableOp_29ขAssignVariableOp_3ขAssignVariableOp_30ขAssignVariableOp_31ขAssignVariableOp_32ขAssignVariableOp_4ขAssignVariableOp_5ขAssignVariableOp_6ขAssignVariableOp_7ขAssignVariableOp_8ขAssignVariableOp_9ฒ
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*ุ
valueฮBห"B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHด
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ห
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapes
::::::::::::::::::::::::::::::::::*0
dtypes&
$2"	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp assignvariableop_conv2d_4_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv2d_4_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv2d_5_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv2d_5_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_4_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_4_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_5_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_5_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_iterIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_beta_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_beta_2Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_decayIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOp&assignvariableop_12_adam_learning_rateIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOpassignvariableop_13_total_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOpassignvariableop_14_count_1Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOpassignvariableop_15_totalIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOpassignvariableop_16_countIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOp*assignvariableop_17_adam_conv2d_4_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOp(assignvariableop_18_adam_conv2d_4_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp*assignvariableop_19_adam_conv2d_5_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOp(assignvariableop_20_adam_conv2d_5_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp)assignvariableop_21_adam_dense_4_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp'assignvariableop_22_adam_dense_4_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOp)assignvariableop_23_adam_dense_5_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOp'assignvariableop_24_adam_dense_5_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_conv2d_4_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_conv2d_4_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOp*assignvariableop_27_adam_conv2d_5_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_28AssignVariableOp(assignvariableop_28_adam_conv2d_5_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_29AssignVariableOp)assignvariableop_29_adam_dense_4_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_30AssignVariableOp'assignvariableop_30_adam_dense_4_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_31AssignVariableOp)assignvariableop_31_adam_dense_5_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_32AssignVariableOp'assignvariableop_32_adam_dense_5_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ฅ
Identity_33Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_34IdentityIdentity_33:output:0^NoOp_1*
T0*
_output_shapes
: 
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_34Identity_34:output:0*W
_input_shapesF
D: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
๎
b
F__inference_activation_4_layer_call_and_return_conditional_losses_6386

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:?????????bbศc
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????bbศ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????bbศ:X T
0
_output_shapes
:?????????bbศ
 
_user_specified_nameinputs"ฟL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*ภ
serving_defaultฌ
Q
conv2d_4_input?
 serving_default_conv2d_4_input:0?????????dd;
dense_50
StatefulPartitionedCall:0?????????tensorflow/serving/predict:ื๑
ะ
layer_with_weights-0
layer-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer-6
layer-7
	layer_with_weights-2
	layer-8

layer_with_weights-3

layer-9
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
?
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias
 _jit_compiled_convolution_op"
_tf_keras_layer
ฅ
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses"
_tf_keras_layer
ฅ
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses"
_tf_keras_layer
?
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses

/kernel
0bias
 1_jit_compiled_convolution_op"
_tf_keras_layer
ฅ
2	variables
3trainable_variables
4regularization_losses
5	keras_api
6__call__
*7&call_and_return_all_conditional_losses"
_tf_keras_layer
ฅ
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses"
_tf_keras_layer
ฅ
>	variables
?trainable_variables
@regularization_losses
A	keras_api
B__call__
*C&call_and_return_all_conditional_losses"
_tf_keras_layer
ผ
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
H__call__
*I&call_and_return_all_conditional_losses
J_random_generator"
_tf_keras_layer
ป
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
O__call__
*P&call_and_return_all_conditional_losses

Qkernel
Rbias"
_tf_keras_layer
ป
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W__call__
*X&call_and_return_all_conditional_losses

Ykernel
Zbias"
_tf_keras_layer
X
0
1
/2
03
Q4
R5
Y6
Z7"
trackable_list_wrapper
X
0
1
/2
03
Q4
R5
Y6
Z7"
trackable_list_wrapper
 "
trackable_list_wrapper
ส
[non_trainable_variables

\layers
]metrics
^layer_regularization_losses
_layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
โ
`trace_0
atrace_1
btrace_2
ctrace_32๗
+__inference_sequential_2_layer_call_fn_5971
+__inference_sequential_2_layer_call_fn_6255
+__inference_sequential_2_layer_call_fn_6276
+__inference_sequential_2_layer_call_fn_6145ภ
ทฒณ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 z`trace_0zatrace_1zbtrace_2zctrace_3
ฮ
dtrace_0
etrace_1
ftrace_2
gtrace_32ใ
F__inference_sequential_2_layer_call_and_return_conditional_losses_6313
F__inference_sequential_2_layer_call_and_return_conditional_losses_6357
F__inference_sequential_2_layer_call_and_return_conditional_losses_6175
F__inference_sequential_2_layer_call_and_return_conditional_losses_6205ภ
ทฒณ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 zdtrace_0zetrace_1zftrace_2zgtrace_3
ัBฮ
__inference__wrapped_model_5823conv2d_4_input"
ฒ
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
๓
hiter

ibeta_1

jbeta_2
	kdecay
llearning_ratemมmย/mร0mฤQmลRmฦYmวZmศvษvส/vห0vฬQvอRvฮYvฯZvะ"
	optimizer
,
mserving_default"
signature_map
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
ญ
nnon_trainable_variables

olayers
pmetrics
qlayer_regularization_losses
rlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
๋
strace_02ฮ
'__inference_conv2d_4_layer_call_fn_6366ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 zstrace_0

ttrace_02้
B__inference_conv2d_4_layer_call_and_return_conditional_losses_6376ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 zttrace_0
*:(ศ2conv2d_4/kernel
:ศ2conv2d_4/bias
ด2ฑฎ
ฃฒ
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ญ
unon_trainable_variables

vlayers
wmetrics
xlayer_regularization_losses
ylayer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses"
_generic_user_object
๏
ztrace_02า
+__inference_activation_4_layer_call_fn_6381ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 zztrace_0

{trace_02ํ
F__inference_activation_4_layer_call_and_return_conditional_losses_6386ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 z{trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ฎ
|non_trainable_variables

}layers
~metrics
layer_regularization_losses
layer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses"
_generic_user_object
๔
trace_02ี
.__inference_max_pooling2d_4_layer_call_fn_6391ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 ztrace_0

trace_02๐
I__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_6396ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 ztrace_0
.
/0
01"
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
 "
trackable_list_wrapper
ฒ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses"
_generic_user_object
ํ
trace_02ฮ
'__inference_conv2d_5_layer_call_fn_6405ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 ztrace_0

trace_02้
B__inference_conv2d_5_layer_call_and_return_conditional_losses_6415ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 ztrace_0
*:(ศd2conv2d_5/kernel
:d2conv2d_5/bias
ด2ฑฎ
ฃฒ
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ฒ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
2	variables
3trainable_variables
4regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses"
_generic_user_object
๑
trace_02า
+__inference_activation_5_layer_call_fn_6420ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 ztrace_0

trace_02ํ
F__inference_activation_5_layer_call_and_return_conditional_losses_6425ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 ztrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ฒ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses"
_generic_user_object
๔
trace_02ี
.__inference_max_pooling2d_5_layer_call_fn_6430ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 ztrace_0

trace_02๐
I__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_6435ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 ztrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ฒ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
>	variables
?trainable_variables
@regularization_losses
B__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses"
_generic_user_object
๎
trace_02ฯ
(__inference_flatten_2_layer_call_fn_6440ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 ztrace_0

trace_02๊
C__inference_flatten_2_layer_call_and_return_conditional_losses_6446ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 ztrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ฒ
non_trainable_variables
?layers
กmetrics
 ขlayer_regularization_losses
ฃlayer_metrics
D	variables
Etrainable_variables
Fregularization_losses
H__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses"
_generic_user_object
ฦ
คtrace_0
ฅtrace_12
(__inference_dropout_2_layer_call_fn_6451
(__inference_dropout_2_layer_call_fn_6456ด
ซฒง
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 zคtrace_0zฅtrace_1
?
ฆtrace_0
งtrace_12ม
C__inference_dropout_2_layer_call_and_return_conditional_losses_6461
C__inference_dropout_2_layer_call_and_return_conditional_losses_6473ด
ซฒง
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 zฆtrace_0zงtrace_1
"
_generic_user_object
.
Q0
R1"
trackable_list_wrapper
.
Q0
R1"
trackable_list_wrapper
 "
trackable_list_wrapper
ฒ
จnon_trainable_variables
ฉlayers
ชmetrics
 ซlayer_regularization_losses
ฌlayer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses"
_generic_user_object
์
ญtrace_02อ
&__inference_dense_4_layer_call_fn_6482ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 zญtrace_0

ฎtrace_02่
A__inference_dense_4_layer_call_and_return_conditional_losses_6493ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 zฎtrace_0
": 
ค22dense_4/kernel
:22dense_4/bias
.
Y0
Z1"
trackable_list_wrapper
.
Y0
Z1"
trackable_list_wrapper
 "
trackable_list_wrapper
ฒ
ฏnon_trainable_variables
ฐlayers
ฑmetrics
 ฒlayer_regularization_losses
ณlayer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses"
_generic_user_object
์
ดtrace_02อ
&__inference_dense_5_layer_call_fn_6502ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 zดtrace_0

ตtrace_02่
A__inference_dense_5_layer_call_and_return_conditional_losses_6513ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 zตtrace_0
 :22dense_5/kernel
:2dense_5/bias
 "
trackable_list_wrapper
f
0
1
2
3
4
5
6
7
	8

9"
trackable_list_wrapper
0
ถ0
ท1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
+__inference_sequential_2_layer_call_fn_5971conv2d_4_input"ภ
ทฒณ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
?B๚
+__inference_sequential_2_layer_call_fn_6255inputs"ภ
ทฒณ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
?B๚
+__inference_sequential_2_layer_call_fn_6276inputs"ภ
ทฒณ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
B
+__inference_sequential_2_layer_call_fn_6145conv2d_4_input"ภ
ทฒณ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
B
F__inference_sequential_2_layer_call_and_return_conditional_losses_6313inputs"ภ
ทฒณ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
B
F__inference_sequential_2_layer_call_and_return_conditional_losses_6357inputs"ภ
ทฒณ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
?B
F__inference_sequential_2_layer_call_and_return_conditional_losses_6175conv2d_4_input"ภ
ทฒณ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
?B
F__inference_sequential_2_layer_call_and_return_conditional_losses_6205conv2d_4_input"ภ
ทฒณ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
ะBอ
"__inference_signature_wrapper_6234conv2d_4_input"
ฒ
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?Bุ
'__inference_conv2d_4_layer_call_fn_6366inputs"ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
๖B๓
B__inference_conv2d_4_layer_call_and_return_conditional_losses_6376inputs"ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
฿B?
+__inference_activation_4_layer_call_fn_6381inputs"ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
๚B๗
F__inference_activation_4_layer_call_and_return_conditional_losses_6386inputs"ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
โB฿
.__inference_max_pooling2d_4_layer_call_fn_6391inputs"ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
?B๚
I__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_6396inputs"ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?Bุ
'__inference_conv2d_5_layer_call_fn_6405inputs"ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
๖B๓
B__inference_conv2d_5_layer_call_and_return_conditional_losses_6415inputs"ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
฿B?
+__inference_activation_5_layer_call_fn_6420inputs"ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
๚B๗
F__inference_activation_5_layer_call_and_return_conditional_losses_6425inputs"ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
โB฿
.__inference_max_pooling2d_5_layer_call_fn_6430inputs"ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
?B๚
I__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_6435inputs"ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?Bู
(__inference_flatten_2_layer_call_fn_6440inputs"ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
๗B๔
C__inference_flatten_2_layer_call_and_return_conditional_losses_6446inputs"ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
๎B๋
(__inference_dropout_2_layer_call_fn_6451inputs"ด
ซฒง
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
๎B๋
(__inference_dropout_2_layer_call_fn_6456inputs"ด
ซฒง
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
B
C__inference_dropout_2_layer_call_and_return_conditional_losses_6461inputs"ด
ซฒง
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
B
C__inference_dropout_2_layer_call_and_return_conditional_losses_6473inputs"ด
ซฒง
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ฺBื
&__inference_dense_4_layer_call_fn_6482inputs"ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
๕B๒
A__inference_dense_4_layer_call_and_return_conditional_losses_6493inputs"ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ฺBื
&__inference_dense_5_layer_call_fn_6502inputs"ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
๕B๒
A__inference_dense_5_layer_call_and_return_conditional_losses_6513inputs"ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
R
ธ	variables
น	keras_api

บtotal

ปcount"
_tf_keras_metric
c
ผ	variables
ฝ	keras_api

พtotal

ฟcount
ภ
_fn_kwargs"
_tf_keras_metric
0
บ0
ป1"
trackable_list_wrapper
.
ธ	variables"
_generic_user_object
:  (2total
:  (2count
0
พ0
ฟ1"
trackable_list_wrapper
.
ผ	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
/:-ศ2Adam/conv2d_4/kernel/m
!:ศ2Adam/conv2d_4/bias/m
/:-ศd2Adam/conv2d_5/kernel/m
 :d2Adam/conv2d_5/bias/m
':%
ค22Adam/dense_4/kernel/m
:22Adam/dense_4/bias/m
%:#22Adam/dense_5/kernel/m
:2Adam/dense_5/bias/m
/:-ศ2Adam/conv2d_4/kernel/v
!:ศ2Adam/conv2d_4/bias/v
/:-ศd2Adam/conv2d_5/kernel/v
 :d2Adam/conv2d_5/bias/v
':%
ค22Adam/dense_4/kernel/v
:22Adam/dense_4/bias/v
%:#22Adam/dense_5/kernel/v
:2Adam/dense_5/bias/vก
__inference__wrapped_model_5823~/0QRYZ?ข<
5ข2
0-
conv2d_4_input?????????dd
ช "1ช.
,
dense_5!
dense_5?????????ด
F__inference_activation_4_layer_call_and_return_conditional_losses_6386j8ข5
.ข+
)&
inputs?????????bbศ
ช ".ข+
$!
0?????????bbศ
 
+__inference_activation_4_layer_call_fn_6381]8ข5
.ข+
)&
inputs?????????bbศ
ช "!?????????bbศฒ
F__inference_activation_5_layer_call_and_return_conditional_losses_6425h7ข4
-ข*
(%
inputs?????????//d
ช "-ข*
# 
0?????????//d
 
+__inference_activation_5_layer_call_fn_6420[7ข4
-ข*
(%
inputs?????????//d
ช " ?????????//dณ
B__inference_conv2d_4_layer_call_and_return_conditional_losses_6376m7ข4
-ข*
(%
inputs?????????dd
ช ".ข+
$!
0?????????bbศ
 
'__inference_conv2d_4_layer_call_fn_6366`7ข4
-ข*
(%
inputs?????????dd
ช "!?????????bbศณ
B__inference_conv2d_5_layer_call_and_return_conditional_losses_6415m/08ข5
.ข+
)&
inputs?????????11ศ
ช "-ข*
# 
0?????????//d
 
'__inference_conv2d_5_layer_call_fn_6405`/08ข5
.ข+
)&
inputs?????????11ศ
ช " ?????????//dฃ
A__inference_dense_4_layer_call_and_return_conditional_losses_6493^QR1ข.
'ข$
"
inputs?????????ค
ช "%ข"

0?????????2
 {
&__inference_dense_4_layer_call_fn_6482QQR1ข.
'ข$
"
inputs?????????ค
ช "?????????2ก
A__inference_dense_5_layer_call_and_return_conditional_losses_6513\YZ/ข,
%ข"
 
inputs?????????2
ช "%ข"

0?????????
 y
&__inference_dense_5_layer_call_fn_6502OYZ/ข,
%ข"
 
inputs?????????2
ช "?????????ง
C__inference_dropout_2_layer_call_and_return_conditional_losses_6461`5ข2
+ข(
"
inputs?????????ค
p 
ช "'ข$

0?????????ค
 ง
C__inference_dropout_2_layer_call_and_return_conditional_losses_6473`5ข2
+ข(
"
inputs?????????ค
p
ช "'ข$

0?????????ค
 
(__inference_dropout_2_layer_call_fn_6451S5ข2
+ข(
"
inputs?????????ค
p 
ช "?????????ค
(__inference_dropout_2_layer_call_fn_6456S5ข2
+ข(
"
inputs?????????ค
p
ช "?????????คฉ
C__inference_flatten_2_layer_call_and_return_conditional_losses_6446b7ข4
-ข*
(%
inputs?????????d
ช "'ข$

0?????????ค
 
(__inference_flatten_2_layer_call_fn_6440U7ข4
-ข*
(%
inputs?????????d
ช "?????????ค์
I__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_6396RขO
HขE
C@
inputs4????????????????????????????????????
ช "HขE
>;
04????????????????????????????????????
 ฤ
.__inference_max_pooling2d_4_layer_call_fn_6391RขO
HขE
C@
inputs4????????????????????????????????????
ช ";84????????????????????????????????????์
I__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_6435RขO
HขE
C@
inputs4????????????????????????????????????
ช "HขE
>;
04????????????????????????????????????
 ฤ
.__inference_max_pooling2d_5_layer_call_fn_6430RขO
HขE
C@
inputs4????????????????????????????????????
ช ";84????????????????????????????????????ฤ
F__inference_sequential_2_layer_call_and_return_conditional_losses_6175z/0QRYZGขD
=ข:
0-
conv2d_4_input?????????dd
p 

 
ช "%ข"

0?????????
 ฤ
F__inference_sequential_2_layer_call_and_return_conditional_losses_6205z/0QRYZGขD
=ข:
0-
conv2d_4_input?????????dd
p

 
ช "%ข"

0?????????
 ผ
F__inference_sequential_2_layer_call_and_return_conditional_losses_6313r/0QRYZ?ข<
5ข2
(%
inputs?????????dd
p 

 
ช "%ข"

0?????????
 ผ
F__inference_sequential_2_layer_call_and_return_conditional_losses_6357r/0QRYZ?ข<
5ข2
(%
inputs?????????dd
p

 
ช "%ข"

0?????????
 
+__inference_sequential_2_layer_call_fn_5971m/0QRYZGขD
=ข:
0-
conv2d_4_input?????????dd
p 

 
ช "?????????
+__inference_sequential_2_layer_call_fn_6145m/0QRYZGขD
=ข:
0-
conv2d_4_input?????????dd
p

 
ช "?????????
+__inference_sequential_2_layer_call_fn_6255e/0QRYZ?ข<
5ข2
(%
inputs?????????dd
p 

 
ช "?????????
+__inference_sequential_2_layer_call_fn_6276e/0QRYZ?ข<
5ข2
(%
inputs?????????dd
p

 
ช "?????????ท
"__inference_signature_wrapper_6234/0QRYZQขN
ข 
GชD
B
conv2d_4_input0-
conv2d_4_input?????????dd"1ช.
,
dense_5!
dense_5?????????