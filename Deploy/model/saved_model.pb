??
??
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ?
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
?
HashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetype?
.
Identity

input"T
output"T"	
Ttype
w
LookupTableFindV2
table_handle
keys"Tin
default_value"Tout
values"Tout"
Tintype"
Touttype?
b
LookupTableImportV2
table_handle
keys"Tin
values"Tout"
Tintype"
Touttype?
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
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
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
?
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	?
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
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
?
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
executor_typestring ??
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
f
TopKV2

input"T
k
values"T
indices"
sortedbool("
Ttype:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68??
o
identifiersVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameidentifiers
h
identifiers/Read/ReadVariableOpReadVariableOpidentifiers*
_output_shapes	
:?*
dtype0
q

candidatesVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*
shared_name
candidates
j
candidates/Read/ReadVariableOpReadVariableOp
candidates*
_output_shapes
:	?*
dtype0
?
embedding_1/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?)*'
shared_nameembedding_1/embeddings
?
*embedding_1/embeddings/Read/ReadVariableOpReadVariableOpembedding_1/embeddings*
_output_shapes
:	?)*
dtype0
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:)*
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

:)*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:*
dtype0
k

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name303*
value_dtype0	
G
ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R 
?
Const_1Const*
_output_shapes	
:?*
dtype0*?
value?B??B0B1B10B100B101B102B103B104B105B106B107B108B109B11B110B111B112B113B114B115B116B117B118B119B12B120B121B122B123B124B125B126B127B128B129B13B130B131B132B133B134B135B136B137B138B139B14B140B141B142B143B144B145B146B147B148B149B15B150B151B152B153B154B155B156B157B158B159B16B160B161B162B163B164B165B166B167B168B169B17B170B171B172B173B174B175B176B177B178B179B18B180B181B182B183B19B2B20B21B22B23B24B25B26B27B28B29B3B30B31B32B33B34B35B36B37B38B39B4B40B41B42B43B44B45B46B47B48B49B5B50B51B52B53B54B55B56B57B58B59B6B60B61B62B63B64B65B66B67B68B69B7B70B71B72B73B74B75B76B77B78B79B8B80B81B82B83B84B85B86B87B88B89B9B90B91B92B93B94B95B96B97B98B99
?
Const_2Const*
_output_shapes	
:?*
dtype0	*?
value?B?	?"?                                                        	       
                                                                                                                                                                  !       "       #       $       %       &       '       (       )       *       +       ,       -       .       /       0       1       2       3       4       5       6       7       8       9       :       ;       <       =       >       ?       @       A       B       C       D       E       F       G       H       I       J       K       L       M       N       O       P       Q       R       S       T       U       V       W       X       Y       Z       [       \       ]       ^       _       `       a       b       c       d       e       f       g       h       i       j       k       l       m       n       o       p       q       r       s       t       u       v       w       x       y       z       {       |       }       ~              ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       
?
StatefulPartitionedCallStatefulPartitionedCall
hash_tableConst_1Const_2*
Tin
2	*
Tout
2*
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
GPU2*0J 8? *"
fR
__inference_<lambda>_8452
&
NoOpNoOp^StatefulPartitionedCall
?
Const_3Const"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B?
?
query_model
identifiers
_identifiers

candidates
_candidates
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*	&call_and_return_all_conditional_losses

_default_save_signature
query_with_exclusions

signatures*
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
KE
VARIABLE_VALUEidentifiers&identifiers/.ATTRIBUTES/VARIABLE_VALUE*
IC
VARIABLE_VALUE
candidates%candidates/.ATTRIBUTES/VARIABLE_VALUE*
'
0
1
2
3
4*

0
1
2*
* 
?
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__

_default_save_signature
*	&call_and_return_all_conditional_losses
&	"call_and_return_conditional_losses*
* 
* 
* 
* 

serving_default* 
#
lookup_table
 	keras_api* 
?

embeddings
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses*
?

kernel
bias
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses*

0
1
2*

0
1
2*
* 
?
-non_trainable_variables

.layers
/metrics
0layer_regularization_losses
1layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
VP
VARIABLE_VALUEembedding_1/embeddings&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEdense_1/kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEdense_1/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0*
* 
* 
* 
* 
R
2_initializer
3_create_resource
4_initialize
5_destroy_resource* 
* 

0*

0*
* 
?
6non_trainable_variables

7layers
8metrics
9layer_regularization_losses
:layer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses*
* 
* 

0
1*

0
1*
* 
?
;non_trainable_variables

<layers
=metrics
>layer_regularization_losses
?layer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses*
* 
* 
* 

0
1
2*
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
r
serving_default_input_1Placeholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCall_1StatefulPartitionedCallserving_default_input_1
hash_tableConstembedding_1/embeddingsdense_1/kerneldense_1/bias
candidatesidentifiers*
Tin

2	*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????*'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8? *+
f&R$
"__inference_signature_wrapper_8320
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameidentifiers/Read/ReadVariableOpcandidates/Read/ReadVariableOp*embedding_1/embeddings/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOpConst_3*
Tin
	2*
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
GPU2*0J 8? *&
f!R
__inference__traced_save_8496
?
StatefulPartitionedCall_3StatefulPartitionedCallsaver_filenameidentifiers
candidatesembedding_1/embeddingsdense_1/kerneldense_1/bias*
Tin

2*
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
GPU2*0J 8? *)
f$R"
 __inference__traced_restore_8521??
?
?
 __inference__traced_restore_8521
file_prefix+
assignvariableop_identifiers:	?0
assignvariableop_1_candidates:	?<
)assignvariableop_2_embedding_1_embeddings:	?)3
!assignvariableop_3_dense_1_kernel:)-
assignvariableop_4_dense_1_bias:

identity_6??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B&identifiers/.ATTRIBUTES/VARIABLE_VALUEB%candidates/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH|
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*,
_output_shapes
::::::*
dtypes

2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOpassignvariableop_identifiersIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOpassignvariableop_1_candidatesIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp)assignvariableop_2_embedding_1_embeddingsIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp!assignvariableop_3_dense_1_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOpassignvariableop_4_dense_1_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?

Identity_5Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^NoOp"/device:CPU:0*
T0*
_output_shapes
: U

Identity_6IdentityIdentity_5:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4*"
_acd_function_control_output(*
_output_shapes
 "!

identity_6Identity_6:output:0*
_input_shapes
: : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_4:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?

?
*__inference_brute_force_layer_call_fn_8237
queries
unknown
	unknown_0	
	unknown_1:	?)
	unknown_2:)
	unknown_3:
	unknown_4:	?
	unknown_5:	?
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallqueriesunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2	*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????*'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_brute_force_layer_call_and_return_conditional_losses_8105o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:L H
#
_output_shapes
:?????????
!
_user_specified_name	queries:

_output_shapes
: 
?+
?
__inference__wrapped_model_7843
input_1W
Sbrute_force_sequential_1_string_lookup_1_none_lookup_lookuptablefindv2_table_handleX
Tbrute_force_sequential_1_string_lookup_1_none_lookup_lookuptablefindv2_default_value	M
:brute_force_sequential_1_embedding_1_embedding_lookup_7820:	?)Q
?brute_force_sequential_1_dense_1_matmul_readvariableop_resource:)N
@brute_force_sequential_1_dense_1_biasadd_readvariableop_resource:=
*brute_force_matmul_readvariableop_resource:	?*
brute_force_gather_resource:	?
identity

identity_1??brute_force/Gather?!brute_force/MatMul/ReadVariableOp?7brute_force/sequential_1/dense_1/BiasAdd/ReadVariableOp?6brute_force/sequential_1/dense_1/MatMul/ReadVariableOp?5brute_force/sequential_1/embedding_1/embedding_lookup?Fbrute_force/sequential_1/string_lookup_1/None_Lookup/LookupTableFindV2?
Fbrute_force/sequential_1/string_lookup_1/None_Lookup/LookupTableFindV2LookupTableFindV2Sbrute_force_sequential_1_string_lookup_1_none_lookup_lookuptablefindv2_table_handleinput_1Tbrute_force_sequential_1_string_lookup_1_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
1brute_force/sequential_1/string_lookup_1/IdentityIdentityObrute_force/sequential_1/string_lookup_1/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
5brute_force/sequential_1/embedding_1/embedding_lookupResourceGather:brute_force_sequential_1_embedding_1_embedding_lookup_7820:brute_force/sequential_1/string_lookup_1/Identity:output:0*
Tindices0	*M
_classC
A?loc:@brute_force/sequential_1/embedding_1/embedding_lookup/7820*'
_output_shapes
:?????????)*
dtype0?
>brute_force/sequential_1/embedding_1/embedding_lookup/IdentityIdentity>brute_force/sequential_1/embedding_1/embedding_lookup:output:0*
T0*M
_classC
A?loc:@brute_force/sequential_1/embedding_1/embedding_lookup/7820*'
_output_shapes
:?????????)?
@brute_force/sequential_1/embedding_1/embedding_lookup/Identity_1IdentityGbrute_force/sequential_1/embedding_1/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????)?
6brute_force/sequential_1/dense_1/MatMul/ReadVariableOpReadVariableOp?brute_force_sequential_1_dense_1_matmul_readvariableop_resource*
_output_shapes

:)*
dtype0?
'brute_force/sequential_1/dense_1/MatMulMatMulIbrute_force/sequential_1/embedding_1/embedding_lookup/Identity_1:output:0>brute_force/sequential_1/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
7brute_force/sequential_1/dense_1/BiasAdd/ReadVariableOpReadVariableOp@brute_force_sequential_1_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
(brute_force/sequential_1/dense_1/BiasAddBiasAdd1brute_force/sequential_1/dense_1/MatMul:product:0?brute_force/sequential_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
%brute_force/sequential_1/dense_1/ReluRelu1brute_force/sequential_1/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
!brute_force/MatMul/ReadVariableOpReadVariableOp*brute_force_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
brute_force/MatMulMatMul3brute_force/sequential_1/dense_1/Relu:activations:0)brute_force/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????*
transpose_b(V
brute_force/TopKV2/kConst*
_output_shapes
: *
dtype0*
value	B :?
brute_force/TopKV2TopKV2brute_force/MatMul:product:0brute_force/TopKV2/k:output:0*
T0*:
_output_shapes(
&:?????????:??????????
brute_force/GatherResourceGatherbrute_force_gather_resourcebrute_force/TopKV2:indices:0*
Tindices0*'
_output_shapes
:?????????*
dtype0o
brute_force/IdentityIdentitybrute_force/Gather:output:0*
T0*'
_output_shapes
:?????????j
IdentityIdentitybrute_force/TopKV2:values:0^NoOp*
T0*'
_output_shapes
:?????????n

Identity_1Identitybrute_force/Identity:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^brute_force/Gather"^brute_force/MatMul/ReadVariableOp8^brute_force/sequential_1/dense_1/BiasAdd/ReadVariableOp7^brute_force/sequential_1/dense_1/MatMul/ReadVariableOp6^brute_force/sequential_1/embedding_1/embedding_lookupG^brute_force/sequential_1/string_lookup_1/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : : : : : 2(
brute_force/Gatherbrute_force/Gather2F
!brute_force/MatMul/ReadVariableOp!brute_force/MatMul/ReadVariableOp2r
7brute_force/sequential_1/dense_1/BiasAdd/ReadVariableOp7brute_force/sequential_1/dense_1/BiasAdd/ReadVariableOp2p
6brute_force/sequential_1/dense_1/MatMul/ReadVariableOp6brute_force/sequential_1/dense_1/MatMul/ReadVariableOp2n
5brute_force/sequential_1/embedding_1/embedding_lookup5brute_force/sequential_1/embedding_1/embedding_lookup2?
Fbrute_force/sequential_1/string_lookup_1/None_Lookup/LookupTableFindV2Fbrute_force/sequential_1/string_lookup_1/None_Lookup/LookupTableFindV2:L H
#
_output_shapes
:?????????
!
_user_specified_name	input_1:

_output_shapes
: 
?
?
&__inference_dense_1_layer_call_fn_8415

inputs
unknown:)
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_7878o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????): : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????)
 
_user_specified_nameinputs
?

?
"__inference_signature_wrapper_8320
input_1
unknown
	unknown_0	
	unknown_1:	?)
	unknown_2:)
	unknown_3:
	unknown_4:	?
	unknown_5:	?
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2	*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????*'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8? *(
f#R!
__inference__wrapped_model_7843o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:L H
#
_output_shapes
:?????????
!
_user_specified_name	input_1:

_output_shapes
: 
?
?
F__inference_sequential_1_layer_call_and_return_conditional_losses_8370

inputs>
:string_lookup_1_none_lookup_lookuptablefindv2_table_handle?
;string_lookup_1_none_lookup_lookuptablefindv2_default_value	4
!embedding_1_embedding_lookup_8357:	?)8
&dense_1_matmul_readvariableop_resource:)5
'dense_1_biasadd_readvariableop_resource:
identity??dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?embedding_1/embedding_lookup?-string_lookup_1/None_Lookup/LookupTableFindV2?
-string_lookup_1/None_Lookup/LookupTableFindV2LookupTableFindV2:string_lookup_1_none_lookup_lookuptablefindv2_table_handleinputs;string_lookup_1_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
string_lookup_1/IdentityIdentity6string_lookup_1/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
embedding_1/embedding_lookupResourceGather!embedding_1_embedding_lookup_8357!string_lookup_1/Identity:output:0*
Tindices0	*4
_class*
(&loc:@embedding_1/embedding_lookup/8357*'
_output_shapes
:?????????)*
dtype0?
%embedding_1/embedding_lookup/IdentityIdentity%embedding_1/embedding_lookup:output:0*
T0*4
_class*
(&loc:@embedding_1/embedding_lookup/8357*'
_output_shapes
:?????????)?
'embedding_1/embedding_lookup/Identity_1Identity.embedding_1/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????)?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:)*
dtype0?
dense_1/MatMulMatMul0embedding_1/embedding_lookup/Identity_1:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????`
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????i
IdentityIdentitydense_1/Relu:activations:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^embedding_1/embedding_lookup.^string_lookup_1/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:?????????: : : : : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2<
embedding_1/embedding_lookupembedding_1/embedding_lookup2^
-string_lookup_1/None_Lookup/LookupTableFindV2-string_lookup_1/None_Lookup/LookupTableFindV2:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
9
__inference__creator_8431
identity??
hash_tablek

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name303*
value_dtype0	W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: S
NoOpNoOp^hash_table*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
?
?
F__inference_sequential_1_layer_call_and_return_conditional_losses_8009
string_lookup_1_input>
:string_lookup_1_none_lookup_lookuptablefindv2_table_handle?
;string_lookup_1_none_lookup_lookuptablefindv2_default_value	#
embedding_1_8000:	?)
dense_1_8003:)
dense_1_8005:
identity??dense_1/StatefulPartitionedCall?#embedding_1/StatefulPartitionedCall?-string_lookup_1/None_Lookup/LookupTableFindV2?
-string_lookup_1/None_Lookup/LookupTableFindV2LookupTableFindV2:string_lookup_1_none_lookup_lookuptablefindv2_table_handlestring_lookup_1_input;string_lookup_1_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
string_lookup_1/IdentityIdentity6string_lookup_1/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
#embedding_1/StatefulPartitionedCallStatefulPartitionedCall!string_lookup_1/Identity:output:0embedding_1_8000*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????)*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_embedding_1_layer_call_and_return_conditional_losses_7863?
dense_1/StatefulPartitionedCallStatefulPartitionedCall,embedding_1/StatefulPartitionedCall:output:0dense_1_8003dense_1_8005*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_7878w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_1/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall.^string_lookup_1/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:?????????: : : : : 2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall2^
-string_lookup_1/None_Lookup/LookupTableFindV2-string_lookup_1/None_Lookup/LookupTableFindV2:Z V
#
_output_shapes
:?????????
/
_user_specified_namestring_lookup_1_input:

_output_shapes
: 
?
+
__inference__destroyer_8444
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
?
__inference__traced_save_8496
file_prefix*
&savev2_identifiers_read_readvariableop)
%savev2_candidates_read_readvariableop5
1savev2_embedding_1_embeddings_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop
savev2_const_3

identity_1??MergeV2Checkpointsw
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
_temp/part?
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
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B&identifiers/.ATTRIBUTES/VARIABLE_VALUEB%candidates/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHy
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0&savev2_identifiers_read_readvariableop%savev2_candidates_read_readvariableop1savev2_embedding_1_embeddings_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableopsavev2_const_3"/device:CPU:0*
_output_shapes
 *
dtypes

2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
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

identity_1Identity_1:output:0*D
_input_shapes3
1: :?:	?:	?):):: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:!

_output_shapes	
:?:%!

_output_shapes
:	?:%!

_output_shapes
:	?):$ 

_output_shapes

:): 

_output_shapes
::

_output_shapes
: 
?
?
E__inference_brute_force_layer_call_and_return_conditional_losses_8170
input_1
sequential_1_8148
sequential_1_8150	$
sequential_1_8152:	?)#
sequential_1_8154:)
sequential_1_8156:1
matmul_readvariableop_resource:	?
gather_resource:	?

identity_1

identity_2??Gather?MatMul/ReadVariableOp?$sequential_1/StatefulPartitionedCall?
$sequential_1/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_1_8148sequential_1_8150sequential_1_8152sequential_1_8154sequential_1_8156*
Tin

2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_sequential_1_layer_call_and_return_conditional_losses_7885u
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
MatMulMatMul-sequential_1/StatefulPartitionedCall:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????*
transpose_b(J
TopKV2/kConst*
_output_shapes
: *
dtype0*
value	B :z
TopKV2TopKV2MatMul:product:0TopKV2/k:output:0*
T0*:
_output_shapes(
&:?????????:??????????
GatherResourceGathergather_resourceTopKV2:indices:0*
Tindices0*'
_output_shapes
:?????????*
dtype0W
IdentityIdentityGather:output:0*
T0*'
_output_shapes
:?????????`

Identity_1IdentityTopKV2:values:0^NoOp*
T0*'
_output_shapes
:?????????b

Identity_2IdentityIdentity:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^Gather^MatMul/ReadVariableOp%^sequential_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : : : : : 2
GatherGather2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2L
$sequential_1/StatefulPartitionedCall$sequential_1/StatefulPartitionedCall:L H
#
_output_shapes
:?????????
!
_user_specified_name	input_1:

_output_shapes
: 
?
?
+__inference_sequential_1_layer_call_fn_7898
string_lookup_1_input
unknown
	unknown_0	
	unknown_1:	?)
	unknown_2:)
	unknown_3:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallstring_lookup_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_sequential_1_layer_call_and_return_conditional_losses_7885o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:?????????: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
#
_output_shapes
:?????????
/
_user_specified_namestring_lookup_1_input:

_output_shapes
: 
?
?
__inference_<lambda>_84526
2key_value_init302_lookuptableimportv2_table_handle.
*key_value_init302_lookuptableimportv2_keys0
,key_value_init302_lookuptableimportv2_values	
identity??%key_value_init302/LookupTableImportV2?
%key_value_init302/LookupTableImportV2LookupTableImportV22key_value_init302_lookuptableimportv2_table_handle*key_value_init302_lookuptableimportv2_keys,key_value_init302_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: n
NoOpNoOp&^key_value_init302/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: :?:?2N
%key_value_init302/LookupTableImportV2%key_value_init302/LookupTableImportV2:!

_output_shapes	
:?:!

_output_shapes	
:?
?

?
*__inference_brute_force_layer_call_fn_8057
input_1
unknown
	unknown_0	
	unknown_1:	?)
	unknown_2:)
	unknown_3:
	unknown_4:	?
	unknown_5:	?
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2	*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????*'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_brute_force_layer_call_and_return_conditional_losses_8038o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:L H
#
_output_shapes
:?????????
!
_user_specified_name	input_1:

_output_shapes
: 
?
?
E__inference_embedding_1_layer_call_and_return_conditional_losses_8406

inputs	(
embedding_lookup_8400:	?)
identity??embedding_lookup?
embedding_lookupResourceGatherembedding_lookup_8400inputs*
Tindices0	*(
_class
loc:@embedding_lookup/8400*'
_output_shapes
:?????????)*
dtype0?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*(
_class
loc:@embedding_lookup/8400*'
_output_shapes
:?????????)}
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????)s
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:?????????)Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
E__inference_brute_force_layer_call_and_return_conditional_losses_8195
input_1
sequential_1_8173
sequential_1_8175	$
sequential_1_8177:	?)#
sequential_1_8179:)
sequential_1_8181:1
matmul_readvariableop_resource:	?
gather_resource:	?

identity_1

identity_2??Gather?MatMul/ReadVariableOp?$sequential_1/StatefulPartitionedCall?
$sequential_1/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_1_8173sequential_1_8175sequential_1_8177sequential_1_8179sequential_1_8181*
Tin

2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_sequential_1_layer_call_and_return_conditional_losses_7949u
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
MatMulMatMul-sequential_1/StatefulPartitionedCall:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????*
transpose_b(J
TopKV2/kConst*
_output_shapes
: *
dtype0*
value	B :z
TopKV2TopKV2MatMul:product:0TopKV2/k:output:0*
T0*:
_output_shapes(
&:?????????:??????????
GatherResourceGathergather_resourceTopKV2:indices:0*
Tindices0*'
_output_shapes
:?????????*
dtype0W
IdentityIdentityGather:output:0*
T0*'
_output_shapes
:?????????`

Identity_1IdentityTopKV2:values:0^NoOp*
T0*'
_output_shapes
:?????????b

Identity_2IdentityIdentity:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^Gather^MatMul/ReadVariableOp%^sequential_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : : : : : 2
GatherGather2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2L
$sequential_1/StatefulPartitionedCall$sequential_1/StatefulPartitionedCall:L H
#
_output_shapes
:?????????
!
_user_specified_name	input_1:

_output_shapes
: 
?$
?
E__inference_brute_force_layer_call_and_return_conditional_losses_8297
queriesK
Gsequential_1_string_lookup_1_none_lookup_lookuptablefindv2_table_handleL
Hsequential_1_string_lookup_1_none_lookup_lookuptablefindv2_default_value	A
.sequential_1_embedding_1_embedding_lookup_8274:	?)E
3sequential_1_dense_1_matmul_readvariableop_resource:)B
4sequential_1_dense_1_biasadd_readvariableop_resource:1
matmul_readvariableop_resource:	?
gather_resource:	?

identity_1

identity_2??Gather?MatMul/ReadVariableOp?+sequential_1/dense_1/BiasAdd/ReadVariableOp?*sequential_1/dense_1/MatMul/ReadVariableOp?)sequential_1/embedding_1/embedding_lookup?:sequential_1/string_lookup_1/None_Lookup/LookupTableFindV2?
:sequential_1/string_lookup_1/None_Lookup/LookupTableFindV2LookupTableFindV2Gsequential_1_string_lookup_1_none_lookup_lookuptablefindv2_table_handlequeriesHsequential_1_string_lookup_1_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
%sequential_1/string_lookup_1/IdentityIdentityCsequential_1/string_lookup_1/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
)sequential_1/embedding_1/embedding_lookupResourceGather.sequential_1_embedding_1_embedding_lookup_8274.sequential_1/string_lookup_1/Identity:output:0*
Tindices0	*A
_class7
53loc:@sequential_1/embedding_1/embedding_lookup/8274*'
_output_shapes
:?????????)*
dtype0?
2sequential_1/embedding_1/embedding_lookup/IdentityIdentity2sequential_1/embedding_1/embedding_lookup:output:0*
T0*A
_class7
53loc:@sequential_1/embedding_1/embedding_lookup/8274*'
_output_shapes
:?????????)?
4sequential_1/embedding_1/embedding_lookup/Identity_1Identity;sequential_1/embedding_1/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????)?
*sequential_1/dense_1/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_1_matmul_readvariableop_resource*
_output_shapes

:)*
dtype0?
sequential_1/dense_1/MatMulMatMul=sequential_1/embedding_1/embedding_lookup/Identity_1:output:02sequential_1/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
+sequential_1/dense_1/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential_1/dense_1/BiasAddBiasAdd%sequential_1/dense_1/MatMul:product:03sequential_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????z
sequential_1/dense_1/ReluRelu%sequential_1/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????u
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
MatMulMatMul'sequential_1/dense_1/Relu:activations:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????*
transpose_b(J
TopKV2/kConst*
_output_shapes
: *
dtype0*
value	B :z
TopKV2TopKV2MatMul:product:0TopKV2/k:output:0*
T0*:
_output_shapes(
&:?????????:??????????
GatherResourceGathergather_resourceTopKV2:indices:0*
Tindices0*'
_output_shapes
:?????????*
dtype0W
IdentityIdentityGather:output:0*
T0*'
_output_shapes
:?????????`

Identity_1IdentityTopKV2:values:0^NoOp*
T0*'
_output_shapes
:?????????b

Identity_2IdentityIdentity:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^Gather^MatMul/ReadVariableOp,^sequential_1/dense_1/BiasAdd/ReadVariableOp+^sequential_1/dense_1/MatMul/ReadVariableOp*^sequential_1/embedding_1/embedding_lookup;^sequential_1/string_lookup_1/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : : : : : 2
GatherGather2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2Z
+sequential_1/dense_1/BiasAdd/ReadVariableOp+sequential_1/dense_1/BiasAdd/ReadVariableOp2X
*sequential_1/dense_1/MatMul/ReadVariableOp*sequential_1/dense_1/MatMul/ReadVariableOp2V
)sequential_1/embedding_1/embedding_lookup)sequential_1/embedding_1/embedding_lookup2x
:sequential_1/string_lookup_1/None_Lookup/LookupTableFindV2:sequential_1/string_lookup_1/None_Lookup/LookupTableFindV2:L H
#
_output_shapes
:?????????
!
_user_specified_name	queries:

_output_shapes
: 
?
?
F__inference_sequential_1_layer_call_and_return_conditional_losses_7949

inputs>
:string_lookup_1_none_lookup_lookuptablefindv2_table_handle?
;string_lookup_1_none_lookup_lookuptablefindv2_default_value	#
embedding_1_7940:	?)
dense_1_7943:)
dense_1_7945:
identity??dense_1/StatefulPartitionedCall?#embedding_1/StatefulPartitionedCall?-string_lookup_1/None_Lookup/LookupTableFindV2?
-string_lookup_1/None_Lookup/LookupTableFindV2LookupTableFindV2:string_lookup_1_none_lookup_lookuptablefindv2_table_handleinputs;string_lookup_1_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
string_lookup_1/IdentityIdentity6string_lookup_1/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
#embedding_1/StatefulPartitionedCallStatefulPartitionedCall!string_lookup_1/Identity:output:0embedding_1_7940*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????)*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_embedding_1_layer_call_and_return_conditional_losses_7863?
dense_1/StatefulPartitionedCallStatefulPartitionedCall,embedding_1/StatefulPartitionedCall:output:0dense_1_7943dense_1_7945*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_7878w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_1/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall.^string_lookup_1/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:?????????: : : : : 2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall2^
-string_lookup_1/None_Lookup/LookupTableFindV2-string_lookup_1/None_Lookup/LookupTableFindV2:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?

?
A__inference_dense_1_layer_call_and_return_conditional_losses_7878

inputs0
matmul_readvariableop_resource:)-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:)*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????): : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????)
 
_user_specified_nameinputs
?
?
F__inference_sequential_1_layer_call_and_return_conditional_losses_7993
string_lookup_1_input>
:string_lookup_1_none_lookup_lookuptablefindv2_table_handle?
;string_lookup_1_none_lookup_lookuptablefindv2_default_value	#
embedding_1_7984:	?)
dense_1_7987:)
dense_1_7989:
identity??dense_1/StatefulPartitionedCall?#embedding_1/StatefulPartitionedCall?-string_lookup_1/None_Lookup/LookupTableFindV2?
-string_lookup_1/None_Lookup/LookupTableFindV2LookupTableFindV2:string_lookup_1_none_lookup_lookuptablefindv2_table_handlestring_lookup_1_input;string_lookup_1_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
string_lookup_1/IdentityIdentity6string_lookup_1/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
#embedding_1/StatefulPartitionedCallStatefulPartitionedCall!string_lookup_1/Identity:output:0embedding_1_7984*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????)*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_embedding_1_layer_call_and_return_conditional_losses_7863?
dense_1/StatefulPartitionedCallStatefulPartitionedCall,embedding_1/StatefulPartitionedCall:output:0dense_1_7987dense_1_7989*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_7878w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_1/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall.^string_lookup_1/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:?????????: : : : : 2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall2^
-string_lookup_1/None_Lookup/LookupTableFindV2-string_lookup_1/None_Lookup/LookupTableFindV2:Z V
#
_output_shapes
:?????????
/
_user_specified_namestring_lookup_1_input:

_output_shapes
: 
?

?
*__inference_brute_force_layer_call_fn_8216
queries
unknown
	unknown_0	
	unknown_1:	?)
	unknown_2:)
	unknown_3:
	unknown_4:	?
	unknown_5:	?
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallqueriesunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2	*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????*'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_brute_force_layer_call_and_return_conditional_losses_8038o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:L H
#
_output_shapes
:?????????
!
_user_specified_name	queries:

_output_shapes
: 
?
?
E__inference_embedding_1_layer_call_and_return_conditional_losses_7863

inputs	(
embedding_lookup_7857:	?)
identity??embedding_lookup?
embedding_lookupResourceGatherembedding_lookup_7857inputs*
Tindices0	*(
_class
loc:@embedding_lookup/7857*'
_output_shapes
:?????????)*
dtype0?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*(
_class
loc:@embedding_lookup/7857*'
_output_shapes
:?????????)}
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????)s
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:?????????)Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
A__inference_dense_1_layer_call_and_return_conditional_losses_8426

inputs0
matmul_readvariableop_resource:)-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:)*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????): : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????)
 
_user_specified_nameinputs
?
?
E__inference_brute_force_layer_call_and_return_conditional_losses_8038
queries
sequential_1_8016
sequential_1_8018	$
sequential_1_8020:	?)#
sequential_1_8022:)
sequential_1_8024:1
matmul_readvariableop_resource:	?
gather_resource:	?

identity_1

identity_2??Gather?MatMul/ReadVariableOp?$sequential_1/StatefulPartitionedCall?
$sequential_1/StatefulPartitionedCallStatefulPartitionedCallqueriessequential_1_8016sequential_1_8018sequential_1_8020sequential_1_8022sequential_1_8024*
Tin

2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_sequential_1_layer_call_and_return_conditional_losses_7885u
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
MatMulMatMul-sequential_1/StatefulPartitionedCall:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????*
transpose_b(J
TopKV2/kConst*
_output_shapes
: *
dtype0*
value	B :z
TopKV2TopKV2MatMul:product:0TopKV2/k:output:0*
T0*:
_output_shapes(
&:?????????:??????????
GatherResourceGathergather_resourceTopKV2:indices:0*
Tindices0*'
_output_shapes
:?????????*
dtype0W
IdentityIdentityGather:output:0*
T0*'
_output_shapes
:?????????`

Identity_1IdentityTopKV2:values:0^NoOp*
T0*'
_output_shapes
:?????????b

Identity_2IdentityIdentity:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^Gather^MatMul/ReadVariableOp%^sequential_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : : : : : 2
GatherGather2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2L
$sequential_1/StatefulPartitionedCall$sequential_1/StatefulPartitionedCall:L H
#
_output_shapes
:?????????
!
_user_specified_name	queries:

_output_shapes
: 
?

*__inference_embedding_1_layer_call_fn_8397

inputs	
unknown:	?)
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????)*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_embedding_1_layer_call_and_return_conditional_losses_7863o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????)`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
+__inference_sequential_1_layer_call_fn_7977
string_lookup_1_input
unknown
	unknown_0	
	unknown_1:	?)
	unknown_2:)
	unknown_3:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallstring_lookup_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_sequential_1_layer_call_and_return_conditional_losses_7949o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:?????????: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
#
_output_shapes
:?????????
/
_user_specified_namestring_lookup_1_input:

_output_shapes
: 
?
?
__inference__initializer_84396
2key_value_init302_lookuptableimportv2_table_handle.
*key_value_init302_lookuptableimportv2_keys0
,key_value_init302_lookuptableimportv2_values	
identity??%key_value_init302/LookupTableImportV2?
%key_value_init302/LookupTableImportV2LookupTableImportV22key_value_init302_lookuptableimportv2_table_handle*key_value_init302_lookuptableimportv2_keys,key_value_init302_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: n
NoOpNoOp&^key_value_init302/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: :?:?2N
%key_value_init302/LookupTableImportV2%key_value_init302/LookupTableImportV2:!

_output_shapes	
:?:!

_output_shapes	
:?
?
?
F__inference_sequential_1_layer_call_and_return_conditional_losses_8390

inputs>
:string_lookup_1_none_lookup_lookuptablefindv2_table_handle?
;string_lookup_1_none_lookup_lookuptablefindv2_default_value	4
!embedding_1_embedding_lookup_8377:	?)8
&dense_1_matmul_readvariableop_resource:)5
'dense_1_biasadd_readvariableop_resource:
identity??dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?embedding_1/embedding_lookup?-string_lookup_1/None_Lookup/LookupTableFindV2?
-string_lookup_1/None_Lookup/LookupTableFindV2LookupTableFindV2:string_lookup_1_none_lookup_lookuptablefindv2_table_handleinputs;string_lookup_1_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
string_lookup_1/IdentityIdentity6string_lookup_1/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
embedding_1/embedding_lookupResourceGather!embedding_1_embedding_lookup_8377!string_lookup_1/Identity:output:0*
Tindices0	*4
_class*
(&loc:@embedding_1/embedding_lookup/8377*'
_output_shapes
:?????????)*
dtype0?
%embedding_1/embedding_lookup/IdentityIdentity%embedding_1/embedding_lookup:output:0*
T0*4
_class*
(&loc:@embedding_1/embedding_lookup/8377*'
_output_shapes
:?????????)?
'embedding_1/embedding_lookup/Identity_1Identity.embedding_1/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????)?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:)*
dtype0?
dense_1/MatMulMatMul0embedding_1/embedding_lookup/Identity_1:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????`
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????i
IdentityIdentitydense_1/Relu:activations:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^embedding_1/embedding_lookup.^string_lookup_1/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:?????????: : : : : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2<
embedding_1/embedding_lookupembedding_1/embedding_lookup2^
-string_lookup_1/None_Lookup/LookupTableFindV2-string_lookup_1/None_Lookup/LookupTableFindV2:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?$
?
E__inference_brute_force_layer_call_and_return_conditional_losses_8267
queriesK
Gsequential_1_string_lookup_1_none_lookup_lookuptablefindv2_table_handleL
Hsequential_1_string_lookup_1_none_lookup_lookuptablefindv2_default_value	A
.sequential_1_embedding_1_embedding_lookup_8244:	?)E
3sequential_1_dense_1_matmul_readvariableop_resource:)B
4sequential_1_dense_1_biasadd_readvariableop_resource:1
matmul_readvariableop_resource:	?
gather_resource:	?

identity_1

identity_2??Gather?MatMul/ReadVariableOp?+sequential_1/dense_1/BiasAdd/ReadVariableOp?*sequential_1/dense_1/MatMul/ReadVariableOp?)sequential_1/embedding_1/embedding_lookup?:sequential_1/string_lookup_1/None_Lookup/LookupTableFindV2?
:sequential_1/string_lookup_1/None_Lookup/LookupTableFindV2LookupTableFindV2Gsequential_1_string_lookup_1_none_lookup_lookuptablefindv2_table_handlequeriesHsequential_1_string_lookup_1_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
%sequential_1/string_lookup_1/IdentityIdentityCsequential_1/string_lookup_1/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
)sequential_1/embedding_1/embedding_lookupResourceGather.sequential_1_embedding_1_embedding_lookup_8244.sequential_1/string_lookup_1/Identity:output:0*
Tindices0	*A
_class7
53loc:@sequential_1/embedding_1/embedding_lookup/8244*'
_output_shapes
:?????????)*
dtype0?
2sequential_1/embedding_1/embedding_lookup/IdentityIdentity2sequential_1/embedding_1/embedding_lookup:output:0*
T0*A
_class7
53loc:@sequential_1/embedding_1/embedding_lookup/8244*'
_output_shapes
:?????????)?
4sequential_1/embedding_1/embedding_lookup/Identity_1Identity;sequential_1/embedding_1/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????)?
*sequential_1/dense_1/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_1_matmul_readvariableop_resource*
_output_shapes

:)*
dtype0?
sequential_1/dense_1/MatMulMatMul=sequential_1/embedding_1/embedding_lookup/Identity_1:output:02sequential_1/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
+sequential_1/dense_1/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential_1/dense_1/BiasAddBiasAdd%sequential_1/dense_1/MatMul:product:03sequential_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????z
sequential_1/dense_1/ReluRelu%sequential_1/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????u
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
MatMulMatMul'sequential_1/dense_1/Relu:activations:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????*
transpose_b(J
TopKV2/kConst*
_output_shapes
: *
dtype0*
value	B :z
TopKV2TopKV2MatMul:product:0TopKV2/k:output:0*
T0*:
_output_shapes(
&:?????????:??????????
GatherResourceGathergather_resourceTopKV2:indices:0*
Tindices0*'
_output_shapes
:?????????*
dtype0W
IdentityIdentityGather:output:0*
T0*'
_output_shapes
:?????????`

Identity_1IdentityTopKV2:values:0^NoOp*
T0*'
_output_shapes
:?????????b

Identity_2IdentityIdentity:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^Gather^MatMul/ReadVariableOp,^sequential_1/dense_1/BiasAdd/ReadVariableOp+^sequential_1/dense_1/MatMul/ReadVariableOp*^sequential_1/embedding_1/embedding_lookup;^sequential_1/string_lookup_1/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : : : : : 2
GatherGather2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2Z
+sequential_1/dense_1/BiasAdd/ReadVariableOp+sequential_1/dense_1/BiasAdd/ReadVariableOp2X
*sequential_1/dense_1/MatMul/ReadVariableOp*sequential_1/dense_1/MatMul/ReadVariableOp2V
)sequential_1/embedding_1/embedding_lookup)sequential_1/embedding_1/embedding_lookup2x
:sequential_1/string_lookup_1/None_Lookup/LookupTableFindV2:sequential_1/string_lookup_1/None_Lookup/LookupTableFindV2:L H
#
_output_shapes
:?????????
!
_user_specified_name	queries:

_output_shapes
: 
?
?
E__inference_brute_force_layer_call_and_return_conditional_losses_8105
queries
sequential_1_8083
sequential_1_8085	$
sequential_1_8087:	?)#
sequential_1_8089:)
sequential_1_8091:1
matmul_readvariableop_resource:	?
gather_resource:	?

identity_1

identity_2??Gather?MatMul/ReadVariableOp?$sequential_1/StatefulPartitionedCall?
$sequential_1/StatefulPartitionedCallStatefulPartitionedCallqueriessequential_1_8083sequential_1_8085sequential_1_8087sequential_1_8089sequential_1_8091*
Tin

2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_sequential_1_layer_call_and_return_conditional_losses_7949u
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
MatMulMatMul-sequential_1/StatefulPartitionedCall:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????*
transpose_b(J
TopKV2/kConst*
_output_shapes
: *
dtype0*
value	B :z
TopKV2TopKV2MatMul:product:0TopKV2/k:output:0*
T0*:
_output_shapes(
&:?????????:??????????
GatherResourceGathergather_resourceTopKV2:indices:0*
Tindices0*'
_output_shapes
:?????????*
dtype0W
IdentityIdentityGather:output:0*
T0*'
_output_shapes
:?????????`

Identity_1IdentityTopKV2:values:0^NoOp*
T0*'
_output_shapes
:?????????b

Identity_2IdentityIdentity:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^Gather^MatMul/ReadVariableOp%^sequential_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : : : : : 2
GatherGather2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2L
$sequential_1/StatefulPartitionedCall$sequential_1/StatefulPartitionedCall:L H
#
_output_shapes
:?????????
!
_user_specified_name	queries:

_output_shapes
: 
?
?
+__inference_sequential_1_layer_call_fn_8350

inputs
unknown
	unknown_0	
	unknown_1:	?)
	unknown_2:)
	unknown_3:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_sequential_1_layer_call_and_return_conditional_losses_7949o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:?????????: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
?
+__inference_sequential_1_layer_call_fn_8335

inputs
unknown
	unknown_0	
	unknown_1:	?)
	unknown_2:)
	unknown_3:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_sequential_1_layer_call_and_return_conditional_losses_7885o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:?????????: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?

?
*__inference_brute_force_layer_call_fn_8145
input_1
unknown
	unknown_0	
	unknown_1:	?)
	unknown_2:)
	unknown_3:
	unknown_4:	?
	unknown_5:	?
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2	*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????*'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_brute_force_layer_call_and_return_conditional_losses_8105o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:L H
#
_output_shapes
:?????????
!
_user_specified_name	input_1:

_output_shapes
: 
?
?
F__inference_sequential_1_layer_call_and_return_conditional_losses_7885

inputs>
:string_lookup_1_none_lookup_lookuptablefindv2_table_handle?
;string_lookup_1_none_lookup_lookuptablefindv2_default_value	#
embedding_1_7864:	?)
dense_1_7879:)
dense_1_7881:
identity??dense_1/StatefulPartitionedCall?#embedding_1/StatefulPartitionedCall?-string_lookup_1/None_Lookup/LookupTableFindV2?
-string_lookup_1/None_Lookup/LookupTableFindV2LookupTableFindV2:string_lookup_1_none_lookup_lookuptablefindv2_table_handleinputs;string_lookup_1_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
string_lookup_1/IdentityIdentity6string_lookup_1/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
#embedding_1/StatefulPartitionedCallStatefulPartitionedCall!string_lookup_1/Identity:output:0embedding_1_7864*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????)*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_embedding_1_layer_call_and_return_conditional_losses_7863?
dense_1/StatefulPartitionedCallStatefulPartitionedCall,embedding_1/StatefulPartitionedCall:output:0dense_1_7879dense_1_7881*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_7878w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_1/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall.^string_lookup_1/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:?????????: : : : : 2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall2^
-string_lookup_1/None_Lookup/LookupTableFindV2-string_lookup_1/None_Lookup/LookupTableFindV2:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: "?L
saver_filename:0StatefulPartitionedCall_2:0StatefulPartitionedCall_38"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
7
input_1,
serving_default_input_1:0?????????>
output_12
StatefulPartitionedCall_1:0?????????>
output_22
StatefulPartitionedCall_1:1?????????tensorflow/serving/predict:?[
?
query_model
identifiers
_identifiers

candidates
_candidates
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*	&call_and_return_all_conditional_losses

_default_save_signature
query_with_exclusions

signatures"
_tf_keras_model
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_sequential
:?2identifiers
:	?2
candidates
C
0
1
2
3
4"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
?
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__

_default_save_signature
*	&call_and_return_all_conditional_losses
&	"call_and_return_conditional_losses"
_generic_user_object
?2?
*__inference_brute_force_layer_call_fn_8057
*__inference_brute_force_layer_call_fn_8216
*__inference_brute_force_layer_call_fn_8237
*__inference_brute_force_layer_call_fn_8145?
???
FullArgSpec/
args'?$
jself
	jqueries
jk

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_brute_force_layer_call_and_return_conditional_losses_8267
E__inference_brute_force_layer_call_and_return_conditional_losses_8297
E__inference_brute_force_layer_call_and_return_conditional_losses_8170
E__inference_brute_force_layer_call_and_return_conditional_losses_8195?
???
FullArgSpec/
args'?$
jself
	jqueries
jk

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
__inference__wrapped_model_7843input_1"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec1
args)?&
jself
	jqueries
j
exclusions
jk
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
,
serving_default"
signature_map
:
lookup_table
 	keras_api"
_tf_keras_layer
?

embeddings
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses"
_tf_keras_layer
?

kernel
bias
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses"
_tf_keras_layer
5
0
1
2"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
?
-non_trainable_variables

.layers
/metrics
0layer_regularization_losses
1layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?2?
+__inference_sequential_1_layer_call_fn_7898
+__inference_sequential_1_layer_call_fn_8335
+__inference_sequential_1_layer_call_fn_8350
+__inference_sequential_1_layer_call_fn_7977?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_sequential_1_layer_call_and_return_conditional_losses_8370
F__inference_sequential_1_layer_call_and_return_conditional_losses_8390
F__inference_sequential_1_layer_call_and_return_conditional_losses_7993
F__inference_sequential_1_layer_call_and_return_conditional_losses_8009?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
):'	?)2embedding_1/embeddings
 :)2dense_1/kernel
:2dense_1/bias
.
0
1"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
"__inference_signature_wrapper_8320input_1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
j
2_initializer
3_create_resource
4_initialize
5_destroy_resourceR jCustom.StaticHashTable
"
_generic_user_object
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
6non_trainable_variables

7layers
8metrics
9layer_regularization_losses
:layer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses"
_generic_user_object
?2?
*__inference_embedding_1_layer_call_fn_8397?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_embedding_1_layer_call_and_return_conditional_losses_8406?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
;non_trainable_variables

<layers
=metrics
>layer_regularization_losses
?layer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses"
_generic_user_object
?2?
&__inference_dense_1_layer_call_fn_8415?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_dense_1_layer_call_and_return_conditional_losses_8426?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
"
_generic_user_object
?2?
__inference__creator_8431?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__initializer_8439?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__destroyer_8444?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
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
	J
Const
J	
Const_1
J	
Const_25
__inference__creator_8431?

? 
? "? 7
__inference__destroyer_8444?

? 
? "? >
__inference__initializer_8439AB?

? 
? "? ?
__inference__wrapped_model_7843?@,?)
"?
?
input_1?????????
? "c?`
.
output_1"?
output_1?????????
.
output_2"?
output_2??????????
E__inference_brute_force_layer_call_and_return_conditional_losses_8170?@4?1
*?'
?
input_1?????????

 
p 
? "K?H
A?>
?
0/0?????????
?
0/1?????????
? ?
E__inference_brute_force_layer_call_and_return_conditional_losses_8195?@4?1
*?'
?
input_1?????????

 
p
? "K?H
A?>
?
0/0?????????
?
0/1?????????
? ?
E__inference_brute_force_layer_call_and_return_conditional_losses_8267?@4?1
*?'
?
queries?????????

 
p 
? "K?H
A?>
?
0/0?????????
?
0/1?????????
? ?
E__inference_brute_force_layer_call_and_return_conditional_losses_8297?@4?1
*?'
?
queries?????????

 
p
? "K?H
A?>
?
0/0?????????
?
0/1?????????
? ?
*__inference_brute_force_layer_call_fn_8057~@4?1
*?'
?
input_1?????????

 
p 
? "=?:
?
0?????????
?
1??????????
*__inference_brute_force_layer_call_fn_8145~@4?1
*?'
?
input_1?????????

 
p
? "=?:
?
0?????????
?
1??????????
*__inference_brute_force_layer_call_fn_8216~@4?1
*?'
?
queries?????????

 
p 
? "=?:
?
0?????????
?
1??????????
*__inference_brute_force_layer_call_fn_8237~@4?1
*?'
?
queries?????????

 
p
? "=?:
?
0?????????
?
1??????????
A__inference_dense_1_layer_call_and_return_conditional_losses_8426\/?,
%?"
 ?
inputs?????????)
? "%?"
?
0?????????
? y
&__inference_dense_1_layer_call_fn_8415O/?,
%?"
 ?
inputs?????????)
? "???????????
E__inference_embedding_1_layer_call_and_return_conditional_losses_8406W+?(
!?
?
inputs?????????	
? "%?"
?
0?????????)
? x
*__inference_embedding_1_layer_call_fn_8397J+?(
!?
?
inputs?????????	
? "??????????)?
F__inference_sequential_1_layer_call_and_return_conditional_losses_7993r@B??
8?5
+?(
string_lookup_1_input?????????
p 

 
? "%?"
?
0?????????
? ?
F__inference_sequential_1_layer_call_and_return_conditional_losses_8009r@B??
8?5
+?(
string_lookup_1_input?????????
p

 
? "%?"
?
0?????????
? ?
F__inference_sequential_1_layer_call_and_return_conditional_losses_8370c@3?0
)?&
?
inputs?????????
p 

 
? "%?"
?
0?????????
? ?
F__inference_sequential_1_layer_call_and_return_conditional_losses_8390c@3?0
)?&
?
inputs?????????
p

 
? "%?"
?
0?????????
? ?
+__inference_sequential_1_layer_call_fn_7898e@B??
8?5
+?(
string_lookup_1_input?????????
p 

 
? "???????????
+__inference_sequential_1_layer_call_fn_7977e@B??
8?5
+?(
string_lookup_1_input?????????
p

 
? "???????????
+__inference_sequential_1_layer_call_fn_8335V@3?0
)?&
?
inputs?????????
p 

 
? "???????????
+__inference_sequential_1_layer_call_fn_8350V@3?0
)?&
?
inputs?????????
p

 
? "???????????
"__inference_signature_wrapper_8320?@7?4
? 
-?*
(
input_1?
input_1?????????"c?`
.
output_1"?
output_1?????????
.
output_2"?
output_2?????????