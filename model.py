import tensorflow as tf
from tensorflow.python.ops import math_ops
import numpy as np
from utils import global_variable_list


def act_func(input,act_fcn):
    if act_fcn == 'relu':
        act = tf.nn.relu(input)
    elif act_fcn == 'selu':
        act = tf.nn.selu(input)
    return act

# create fully connected layer
def fc_layer(input, size_in, size_out, namei="fc"):
    with tf.variable_scope(namei, reuse=tf.AUTO_REUSE):
        # stdv = 1. / np.sqrt(size_out)
        w = tf.get_variable("W0", [size_in, size_out], initializer=tf.random_uniform_initializer(- 1. / np.sqrt(size_out),  1. / np.sqrt(size_out)))
        b = tf.get_variable("B0", [size_out], initializer=tf.random_uniform_initializer(- 1. / np.sqrt(size_out),  1. / np.sqrt(size_out)))
        global_variable_list[namei+'/W0'] = w
        global_variable_list[namei + '/B0'] = b
        act = tf.matmul(input, w) + b
        return act


# create 2 fully connected layer with relu activation layer between them
def fc_2layer(input, size_in, size_out, act_fcn, namei="f2c" ):
    with tf.variable_scope(namei, reuse=tf.AUTO_REUSE):
        # stdv1 = 1. / np.sqrt(size_in/10)
        # stdv2 = 1. / np.sqrt(size_out)
        w = tf.get_variable("W", [size_in, size_in // 10], initializer=tf.random_uniform_initializer(-1. / np.sqrt(size_in/10), 1. / np.sqrt(size_in/10)))
        b = tf.get_variable("B", [size_in // 10], initializer=tf.random_uniform_initializer(-1. / np.sqrt(size_in/10), 1. / np.sqrt(size_in/10)))  # []
        w2 = tf.get_variable("W2", [size_in // 10, size_out], initializer=tf.random_uniform_initializer(-1. / np.sqrt(size_out), 1. / np.sqrt(size_out)))
        b2 = tf.get_variable("B2", [size_out], initializer=tf.random_uniform_initializer(-1. / np.sqrt(size_out), 1. / np.sqrt(size_out)))
        global_variable_list[namei+'/W'] = w
        global_variable_list[namei + '/B'] = b
        global_variable_list[namei+'/W2'] = w2
        global_variable_list[namei + '/B2'] = b2

        act = tf.matmul(input, w) + b
        act = act_func(act,act_fcn)
        act = tf.matmul(act, w2) + b2
        return act


def MaskedMSE(output, target, lengths):
    with tf.name_scope('MaskedMSE'):
        # check loss of masked sequence
        mask = tf.cast(tf.expand_dims(tf.transpose(tf.sequence_mask(lengths), [1, 0]), 2), dtype=tf.float32)
        remp1 = tf.multiply(output, mask)
        remp2 = tf.multiply(target, mask)
        loss = tf.reduce_sum(math_ops.squared_difference(remp1, remp2))
        
        # loss = tf.losses.mean_squared_error(remp1, remp2)  # check not divideded
        loss = loss / tf.reduce_sum(mask)
        tf.summary.scalar("loss", loss)
        return loss


#extract speaker and data from look up table
class Encoder:
    def __init__(self, opt):
        super(Encoder, self).__init__()
        self.hidden_size = opt.hidden_size
        self.vocabulary_size = opt.vocabulary_size
        self.nspk = opt.nspk

    def forward(self, input, speakers):
        with tf.name_scope('encoder'):
            W_input = tf.Variable(tf.random_normal([self.vocabulary_size, self.hidden_size]), name="W_encode")
            W_spkr = tf.Variable(tf.random_normal([self.nspk, self.hidden_size]), name="W_spkr")
            global_variable_list['encoder/W_encode'] = W_input
            global_variable_list['encoder/W_spkr'] = W_spkr

            # if isinstance(input, tuple):
            #     lengths = tf.reshape(input[1], [-1])
            #     outputs = tf.nn.embedding_lookup(W_input, input[0], max_norm=1.0)
            #     #      outputs = pack(self.lut_p(input[0]), lengths);
            # else:
            outputs = tf.nn.embedding_lookup(W_input, input, max_norm=1.0)
            #  if isinstance(input, tuple):
            #    outputs = npack(outputs)[0]


            ident = tf.nn.embedding_lookup(W_spkr, speakers, max_norm=1.0)
            ident_Shape=tf.shape(ident)
            ident = tf.squeeze(ident, 1)

            # tf.summary.histogram("W_encode", W_input)
            # tf.summary.histogram("W_spkr", W_spkr)
            # tf.summary.histogram("outputs", outputs)
            # tf.summary.histogram("ident", ident)
            return outputs, ident


class GravesAttention:
    COEF = 0.3989422917366028  # numpy.sqrt(1/(2*numpy.pi))

    def __init__(self, batch_size, mem_elem, K, attention_alignment, act_fcn):
        self.act_fcn = act_fcn
        self.K = K
        self.attention_alignment = attention_alignment
        self.epsilon = 1e-5
        self.batch_si= batch_size
        # self.sm = nn.Softmax()
        self.mem_elem = mem_elem
        # self.N_a = getLinear(mem_elem, 3*K)
        self.J = tf.Variable(tf.tile(tf.reshape(tf.range(0, 500, 1), (1, 1, 500)),
                                     multiples=(batch_size,self.K,1)), trainable=False, name="GA")  #requires_grad=False

    def forward(self, C, context, mu_tm1):
        with tf.name_scope('Graves_Attention'):
            # using dual connected layer to train extraction of gbk_t
            # cclone=C[:self.batch_si]
            cclone = C
            C_Reshaped=tf.reshape(cclone, [self.batch_si,-1], name='c_reshaped')
            gbk_t=fc_2layer(C_Reshaped, self.mem_elem, self.K*3, self.act_fcn, 'f2cGa1')

            #gbk_t_Shape=tf.shape(gbk_t)
            gbk_t = tf.reshape(gbk_t,[self.batch_si, -1, self.K])
            #gbk_t_Shape=tf.shape(gbk_t)
            # gbk_t = tf.Print(gbk_t,[tf.shape(gbk_t), gbk_t, self.batch_si, self.K],'\n- - - SHAPE, VALUE gbk_t:   ')
            # attention model parameters
            g_t = gbk_t[0:self.batch_si, 0, 0:self.K]
            b_t = gbk_t[0:self.batch_si, 1, 0:self.K]
            k_t = gbk_t[0:self.batch_si, 2, 0:self.K]

            # g_t = tf.Print(g_t, [tf.shape(g_t), g_t], '- - - SHAPE, VALUE g_t:   ')
            # b_t = tf.Print(b_t, [tf.shape(b_t), b_t], '- - - SHAPE, VALUE b_t:   ')
            # k_t = tf.Print(k_t, [tf.shape(k_t), k_t], '- - - SHAPE, VALUE k_t:   ')

            # attention GMM parameters
            g_t = tf.nn.softmax(g_t) + self.epsilon
            sig_t = tf.exp(b_t) + self.epsilon
            mu_t =tf.scalar_mul(self.attention_alignment , tf.exp(k_t))+ mu_tm1
            context_Shape=tf.shape(context)
            g_t=tf.tile(tf.expand_dims(g_t,2),multiples=(1,1,context_Shape[1]))
            sig_t =tf.tile(tf.expand_dims(sig_t,2),multiples=(1,1,context_Shape[1]))#
            mu_t_ =tf.tile(tf.expand_dims(mu_t,2),multiples=(1,1,context_Shape[1]))#

            g_t_Shape=tf.shape(g_t)

            J_Shape=tf.shape(self.J)
            j = tf.cast ( tf.slice(self.J,(0,0,0),(g_t_Shape[0], J_Shape[1], context_Shape[1]), name='slice_j')  ,tf.float32)

            # attention weights alpha_t extraction
            phi_t =tf.multiply( g_t , tf.exp(tf.scalar_mul(-0.5 ,tf.multiply( sig_t ,tf.pow(tf.subtract(mu_t_ , j),2)))))
            alpha_t = tf.expand_dims( self.COEF * tf.reduce_sum(phi_t, 1)  ,1)

            # c_t extraction
            c_t = tf.squeeze(tf.transpose(tf.matmul(alpha_t, context),perm=[1, 0, 2]) ,0)

            return c_t, mu_t, alpha_t

# class Attention:
#     def __init__(self, opt, batch_size, mem_elem):
#         self.buff_mem_size = mem_elem
#         self.batch_si = batch_size
#         self.attention_size = opt.attention_size
#         pass
#
#     def forward(self, context, buffer):
#         hidden_size = context.shape[1]
#
#         C_Reshaped = tf.reshape(context, [self.batch_si, -1], name='c_reshaped')
#
#         # Trainable parameters
#         W_omega = tf.Variable(tf.random_normal([self.attention_size, hidden_size], stddev=0.1)) # encodes speaker
#         U_omega = tf.Variable(tf.random_normal([self.attention_size, self.buff_mem_size], stddev=0.1))
#         v_omega = tf.Variable(tf.random_normal([self.attention_size], stddev=0.1))
#
#         with tf.name_scope('Attention Activation'):
#             # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
#             v = tf.tanh()
#
#         alphas = tf.nn.softmax(vu, name='alphas')
#
#         # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
#         output = tf.reduce_sum(context * tf.expand_dims(alphas, -1), 1)
#
#         return output, alphas

class Decoder:
    def __init__(self, opt):
        # our small improvements to the model
        # original model had relu, we change it to selu
        self.act_fcn = opt.act_fcn
        # original model had 2 fully connected layers without activation function in between them
        self.fix_model_flag = opt.fix_model
        # added densenet like connection
        self.improve_model_flag = opt.improve_model

        self.K = opt.K
        self.hidden_size = opt.hidden_size
        self.output_size = opt.output_size

        self.mem_size = opt.mem_size
        self.mem_feat_size = opt.output_size + opt.hidden_size
        self.mem_elem = self.mem_size * self.mem_feat_size
        # self.training=True
        self.optbatch_size=opt.batch_size
        #iniate grave attention parameters
        self.attn = GravesAttention(opt.batch_size,
                                    self.mem_elem,
                                    self.K,
                                    opt.attention_alignment,
                                    self.act_fcn)

        self.mem_feat_size = self.hidden_size + self.output_size

        self.mu_t = tf.Variable(tf.zeros([self.optbatch_size, self.K], dtype=tf.float32), dtype=tf.float32,
                                trainable=False, name="Dmu_t")
        self.S_t = tf.Variable(tf.zeros([self.optbatch_size, self.mem_feat_size, self.mem_size], dtype=tf.float32),
                               dtype=tf.float32, trainable=False, name="DS_t")


    def init_buffer(self, ident, start=True):

        ident_Shape=tf.shape(ident)
        batch_size = ident_Shape[0]
        self.attn.batch_si= batch_size
        assign_opmu_t=tf.assign(self.mu_t[0:batch_size, 0:self.K] , tf.cond(start,
                             lambda: tf.zeros([batch_size, self.K],dtype=ident.dtype),
                             lambda: tf.identity(self.mu_t[0:batch_size, 0:self.K])))
        assign_opS_t=tf.assign( self.S_t[0:batch_size, 0:self.mem_feat_size, 0:self.mem_size] , tf.cond(start,
                             lambda: tf.zeros([batch_size, self.mem_feat_size, self.mem_size],dtype=ident.dtype),
                             lambda: tf.identity(self.S_t[0:batch_size, 0:self.mem_feat_size, 0:self.mem_size])))

        with tf.control_dependencies([assign_opS_t]):
            cond_opS_t = tf.assign(self.S_t[0:batch_size, :self.hidden_size, :],tf.cond(start,
                                 lambda: tf.tile(tf.expand_dims(ident, 2),multiples=(1, 1, self.mem_size)),
                                 lambda: tf.identity(self.S_t[0:batch_size, :self.hidden_size, :])))
        return [cond_opS_t, assign_opmu_t]
        #else:
        #       self.mu_t=tf.remove_training_nodes(self.mu_t)
        #     self.S_t = tf.remove_training_nodes(self.S_t)

    def update_buffer(self, S_tm1, c_t, o_tm1, ident):
        with tf.name_scope('update_buffer'):
            # concat previous output & context
            idt = tf.tanh(fc_layer(ident, self.hidden_size, self.hidden_size, "fcUb1"))
            # :tf.contrib.layers.fully_connected(ident,self.hidden_size))
            o_tm1 = tf.squeeze(o_tm1, [0])
            z_t = tf.concat([c_t + idt, o_tm1 / 30], 1)
            z_t = tf.expand_dims(z_t, 2)
            S_tm1_Shape = tf.shape(S_tm1)
            Sp = tf.concat([z_t, tf.slice(S_tm1, (0, 0, 0), (S_tm1_Shape[0], S_tm1_Shape[1], S_tm1_Shape[2] - 1))], 2)
            Sp_Shape = tf.shape(Sp)
            # update using 2 layers convolution layer
            u = fc_2layer(tf.reshape(Sp, [Sp_Shape[0], -1]), self.mem_elem, self.mem_feat_size, self.act_fcn, namei="f2cBUb1")
            idt_Shape = tf.shape(idt)
            u_Shape = tf.shape(u)
            print(u)

            temp = tf.slice(u, (0, 0), (u_Shape[0], idt_Shape[1])) + idt
            uconcat = tf.concat([temp, u[:, idt_Shape[1]:u_Shape[1]]], 1)
            uconcat = tf.expand_dims(uconcat, 2)

            S_tm1_Shape = tf.shape(S_tm1)
            S = tf.concat([uconcat, tf.slice(S_tm1, (0, 0, 0),(S_tm1_Shape[0], S_tm1_Shape[1], S_tm1_Shape[2]-1))], 2)

            return S

    def while_body(self, idx, o_t, sS_t, mu_t, out, attns,  context, ident, x, train_flag):
        o_tm1 = tf.cond(train_flag,
                        lambda: tf.expand_dims(x[idx]*0.5+o_t*0.5, 0),
                        lambda: tf.expand_dims(o_t, 0)
                        )
        # if not self.training:
        #     o_tm1 = tf.expand_dims(o_t, 0)
        # else:
        #     o_tm1 = tf.expand_dims(x[idx], 0)

        # using trainable grave attention  to extract monotonic c increase in the position along the
        # sequence of input phonemes in the update buffer
        # predict weighted context based on S

        c_t, mu_t, alpha_t = self.attn.forward(sS_t, tf.transpose(context, perm=[1, 0, 2]), mu_t)

        # with tf.control_dependencies([c_t, mu_t, alpha_t]): # add prints to the graph
        #     o_tm1 = tf.Print(o_tm1, [tf.shape(o_tm1)], message='\n SHAPE: o_tm1:  ')
        #     c_t = tf.Print(c_t, [tf.shape(c_t)], message='\n SHAPE: c_t:  ')
        #     mu_t = tf.Print(mu_t, [tf.shape(mu_t)], message='\n SHAPE: mu_t:  ')
        #     alpha_t = tf.Print(alpha_t, [tf.shape(alpha_t)], message='\n SHAPE: alpha_t:  ')
        #     alpha_t = tf.Print(alpha_t, [alpha_t], message='\n --VALUE: alpha_t:  ')
        #     sS_t = tf.Print(sS_t, [tf.shape(sS_t)], message='\n SHAPE: S_t:  ')

        # advance mu and update buffer
        sS_t = self.update_buffer(sS_t, c_t, o_tm1, ident)

        # predict next time step based on buffer content using 2 trainable fully connected layer
        S_t_Shaper = tf.shape(sS_t)
        reshaped_sS_t = tf.reshape(sS_t, (S_t_Shaper[0], -1))
        if self.improve_model_flag:
            new_size = np.round(0.8*self.hidden_size)
        else:
            new_size = self.hidden_size

        ot_out = fc_2layer(reshaped_sS_t, self.mem_elem, new_size, self.act_fcn, namei="f2cb1")
        sp_out = fc_layer(ident, self.hidden_size, new_size, namei="fcb1")
        if self.fix_model_flag:
            ot_out = act_func(ot_out, self.act_fcn)
            sp_out = act_func(sp_out, self.act_fcn)

        if self.improve_model_flag:
            o_t = fc_layer(tf.concat([ot_out, sp_out], 1), 2 * new_size, self.output_size, namei="fcb2")
        else:
            o_t = fc_layer(ot_out + sp_out, self.hidden_size, self.output_size, namei="fcb2")

        # out = tf.concat([out, tf.expand_dims(o_t,0)], axis=0, name='concat_out')
        # attns = tf.concat([attns,tf.expand_dims(tf.squeeze(alpha_t), 0)], axis=0)
        # attns += [tf.squeeze(alpha_t)]
        out = out.write(idx, o_t)
        attns = attns.write(idx, tf.squeeze(alpha_t))

        # with tf.control_dependencies([o_t]):
        #     idx = tf.Print(idx, [idx, o_t, tf.shape(o_t)], message="\n - - idx, o_t, tf.shape(o_t)  :")

        return [idx + 1, o_t, sS_t, mu_t, out, attns,  context, ident, x, train_flag]

    @staticmethod
    def condi(idx, o_t, sS_t, sMu_t, out, attns,  context, ident, x, train_flag):
        x_shape= tf.shape(x)
        return tf.less(idx, x_shape[0])

    def forward(self, x, ident, context, start=True, train_flag=False):
        with tf.name_scope('Decoder'):
            # context_shape = tf.shape(context)
            # out = tf.constant([], tf.float32, shape=(0, self.attn.batch_si, self.output_size))
            # attns = tf.constant([], tf.float32, shape=(0, self.attn.batch_si, context_shape[1]))
            # attns=[1,1]
            o_t = x[0]
            with tf.control_dependencies(self.init_buffer(ident, start)):
                x_shape=tf.shape(x)
                # x = tf.Print(x, [tf.shape(x), x], ' ----- x=:      ')

                out = tf.TensorArray(tf.float32, x_shape[0])
                attns = tf.TensorArray(tf.float32, x_shape[0])

                smu_t = self.mu_t[0:self.attn.batch_si, 0:self.K]
                sS_t = self.S_t[0:self.attn.batch_si]
                idx, o_t, sS_t, smu_t, out, attns,  context, ident, x, _ =\
                   tf.while_loop(self.condi,
                                 self.while_body,
                                 [0, o_t, sS_t, smu_t, out, attns,  context, ident, x, train_flag],
                                 [tf.TensorShape(None), tf.TensorShape(None), tf.TensorShape([None, None, None]),
                                  tf.TensorShape([None, None]), tf.TensorShape(None), tf.TensorShape(None),
                                  tf.TensorShape([None, None, 256]), tf.TensorShape([None, 256]), tf.TensorShape(None),
                                  tf.TensorShape([])])
                assign_op1 = tf.assign(self.mu_t[0:self.attn.batch_si, 0:self.K], smu_t)
                assign_op2 = tf.assign(self.S_t[0:self.attn.batch_si], sS_t)

                out_seq = out.stack()
                attns_seq = attns.stack()
                with tf.control_dependencies([assign_op1, assign_op2]):
                    return out_seq, attns_seq


class Loop:
    def __init__(self, opt):
        # initiate   model parameters of noize output size and decoder parameters
        # decoder shouldn't be initilaized in tf
        self.encoder = Encoder(opt)
        self.decoder = Decoder(opt)
        self.noise = opt.noise
        self.output_size = opt.output_size
        self.training = True

    #initiate input variable
    def init_input(self, tgt, start, train_flag=False):
        with tf.name_scope('init_input'):
            tgt_shaper = tf.shape(tgt)
            with tf.name_scope('init_x_tm1'):
                # x_tm1 is initiated to be zeros(shape=(1,tgt.shape[1],tgt.shape[2])
                # due to tgt being of variable size, we need all this mess :(
                x_tm1 = tf.Variable([], dtype=tf.float32, validate_shape=False, trainable=False)
                zerosop = tf.cond(start,
                                  lambda: tf.zeros([1, tgt_shaper[1], tgt_shaper[2]], dtype=tgt.dtype),
                                  lambda: tf.identity(x_tm1.read_value()))
                assign_op1 = tf.assign(x_tm1, zerosop, validate_shape=False)
            with tf.name_scope('init_inp'):
                with tf.control_dependencies([assign_op1]):
                    inp = tf.cond(tgt_shaper[0] > 1,
                                  lambda: tf.concat([x_tm1.read_value(), tgt[:-1]], 0),
                                  lambda: tf.identity(x_tm1.read_value())
                                  )

                if self.noise > 0:
                    noise = tf.random_normal(tgt_shaper, 0.0, self.noise, dtype=tgt.dtype)
                    inp += noise

                # if not self.training:
                inp = tf.cond(train_flag,
                              lambda: tf.identity(inp),
                              lambda: tf.zeros(tgt_shaper, dtype=inp.dtype)
                              )

                with tf.name_scope('assign_new_x_tm1'):
                    assign_op2 = tf.assign(x_tm1, tf.expand_dims(tgt[-1], 0), validate_shape=False)
                with tf.control_dependencies([assign_op2]):
                    inp = tf.identity(inp)
                return inp


    def forward(self, input0, spkr, tgt, start=True, train_flag=False):
        x = self.init_input(tgt, start, train_flag)
        # get context and indent of current speaker and vocabulary used
        context, ident = self.encoder.forward(input0, spkr)
        # get input after preprocess
        with tf.control_dependencies([x]):
            # get from input and relevant data to speaker and vocabulary the output and attention vector to build the result
            out, attn = self.decoder.forward(x, ident, context, start, train_flag)

            return out, attn
