"""
name:yxh
日期: 2022年07月05日
"""
from rdkit import Chem
import xlrd
from qiskit.providers.aer import AerError
from qiskit import *
from qiskit.circuit import Parameter
from qiskit import execute
import numpy as np
import random
from matplotlib import pyplot as plt
from qiskit.visualization import plot_histogram
from qiskit.algorithms.optimizers import COBYLA, GradientDescent
import multiprocessing
import time
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
b = [0.38740407814041344, 0.6909922173989069, 0.2668461444320309, 0.5195120569929361, 0.9666511706722392, 0.12210590505264074, 0.9149943161570163, 0.48499004246460253, 0.4813613468643514, 0.500923046422819, 0.21779922619068093, 0.5503026787480851, 0.7051627925926517, 0.46821320779745235, 0.19095552475974642, 0.8012761306112315, 0.30864814525806705, 0.52717467737637, 0.018680583147105967, 0.4594300264679766, 0.24845490037074736, 0.14776281196787455, 0.4183609243586529, 0.9059289010347925, 0.06079225789862286, 0.9546756398572285, 0.09384363944264318, 0.96953034875035, 0.07114330460382168, 0.33519518596240094, 0.4336656086414564, 0.09184678810283664, 0.9464095973382165, 0.4642734401981998, 0.7903181998091112, 0.13616375693235516, 0.4451837789080746, 0.23120588018651478, 0.1541021208265544, 0.8449994575237305, 0.15622605190360772, 0.6341649980635233, 0.2826668700992253, 0.26271885811654383, 0.6480568879347429, 0.17568275917307108, 0.6849856677450585, 0.2325167575327992, 0.02959870098639361, 0.11906777877301977, 0.6217993656678362, 0.6749221087398436, 0.30484322214781023, 0.11950759302916303, 0.5985881496788811, 0.40769298171940427, 0.9801419974132902, 0.8468974951436542, 0.7564156881761713, 0.18544659763782678, 0.2334134515341395, 0.2162359347865711, 0.7421323251659959, 0.28661760651129675, 0.19090060769417294, 0.5263825195015431, 0.5510979843810098, 0.6555795613274913, 0.09694542606203704, 0.8331746156330516, 0.22881366685471172, 0.22152406571571603, 0.3724480173877637, 0.9993569068033578, 0.8287021619690064, 0.9568292926279662, 0.33796791312327035, 0.7272268820376813, 0.7262718612615068, 0.45735620387005516, 0.22747317498292186, 0.8899930881042185, 0.475249017062672, 0.47776978618069743, 0.4377819584814887, 0.44822870586450225, 0.6361660361393563, 0.14857544306848047, 0.5916748600554397, 0.03108402759860296, 0.9579999622881478, 0.18677625770366835, 0.6970989906731637, 0.8281033509317011, 0.949255672893882, 0.5721150091346078, 0.2340790104329279, 0.2994981920564954, 0.08730752169927392, 0.3752336177086606, 0.6119317121499964, 0.8102413559920579, 0.7267310781624937, 0.5467953439074532, 0.9315870107279318, 0.5417423901319708, 0.20846255289550875, 0.5877893085708148, 0.08233362805376632, 0.4433867443702926, 0.23695054936142956, 0.33659150766947155, 0.7456601242286104, 0.8199354768933488, 0.3093831264756446, 0.13323630496472805, 0.6994523100006005, 0.25024276993104655, 0.6044920646412999, 0.5381359876933357, 0.4795888604707782, 0.771988640497651, 0.3255732129536293, 0.7712915768046643, 0.06684039474311221, 0.5488277282965274, 0.36029266781774905, 0.10574557633407045, 0.8496052800525778, 0.6648406665693142, 0.537580597030637, 0.8956989832608615, 0.11363533174444762, 0.4646752060142063, 0.6934340863809918, 0.42615888309240757, 0.6679219983167898, 0.3740324734918167, 0.27538411289089826, 0.5759220083022146, 0.7425988502380041, 0.3627318605859794, 0.2923466843173833, 0.5164689977954772, 0.05446162631567375, 0.6352579362502536, 0.4717885754968527, 0.1175741576947491, 0.033802848424643095, 0.4285663631669442, 0.4715635745497293, 0.6045804275622056, 0.7331891191668015, 0.9538485635046261, 0.08390289631207593, 0.48968044351361684, 0.5603324694412585, 0.5543715556651146, 0.8842646696454763, 0.4941632227726862, 0.14033848212881073, 0.9664766030554783, 0.5810428914135631, 0.09099138228339354, 0.8599849953055666, 0.40019162838725686, 0.7597580279233009, 0.6616567943599113, 0.7482094703161266, 0.954839571249887, 0.4044501211518624, 0.07064110894753395, 0.0645938782215667, 0.8969754285800123, 0.560966582460819, 0.8186896958176689, 0.4877617054373531, 0.08459018128419771, 0.6651415244980428, 0.43101627918683305, 0.412446856156656, 0.5715591041407813, 0.09751433724214864, 0.0261448022261056, 0.7948758342465205, 0.5871931354228176, 0.7128022732879756, 0.9762792939479966, 0.8863872896345991, 0.9407858135243561, 0.39435700388483885, 0.18563279398926047, 0.48569583874747624, 0.1473456103132389, 0.07929203096226778, 0.1924479572281732, 0.7549340339875561, 0.9365024528090828, 0.8250510192165573, 0.5805724493724566, 0.7209997779093712, 0.7284450514794633, 0.93320233332565, 0.695574461701561, 0.940830743796286, 0.31422721770254036, 0.4063973486650857, 0.5941489547896478, 0.9213473689062819, 0.22072526905838263, 0.8928878071092383, 0.7111386022903264, 0.8659074346223042, 0.1261540497761252, 0.6643320934015035, 0.415977507976528, 0.835264495393213, 0.9145028402505431, 0.6798213070603943, 0.17269487861071164, 0.21107058340094587, 0.3278067561081003, 0.9871866810277765, 0.5540834889644612, 0.8224265992025905, 0.824840851596547, 0.25311702721122864, 0.5080995642347291, 0.914515151735231, 0.6814576032923543, 0.2773471984541318, 0.7686166783793361, 0.809444458381189, 0.2686505773581288, 0.6845146015199531, 0.9726137180433156, 0.7181333396134364, 0.5732837979669403, 0.22831971279472518, 0.9127894693256208, 0.3450235106067657, 0.18363565381395863, 0.009161971075022923, 0.20030416363798265, 0.41575984272880884, 0.051034995082351875, 0.7961776268102535, 0.6489083084071189, 0.5045467607641037, 0.06264013291782777, 0.27517112272112954, 0.15024893301073394, 0.7122622169379845, 0.4719251106319916, 0.31546166491432837, 0.6825213903656114, 0.384867879319588, 0.44650385890549893, 0.943485921420107, 0.013168647116149623, 0.29329067180846125, 0.6362053911668814, 0.7367039732069065, 0.2406016463313344, 0.5654191272112544, 0.027534717959512456, 0.2620828059673874, 0.6425036996157566, 0.05068403348403716, 0.24710412089199763, 0.8749305758926668, 0.7004309150350277, 0.48293538657493795, 0.31396156511924367, 0.04582067780904797, 0.17209380333698776, 0.4789142750930102, 0.8123228751102448, 0.5516981985048224, 0.3522989822396735, 0.13429688178193788, 0.3121564012691578, 0.30430929285258645, 0.7108562378887047, 0.924693204370302, 0.32280641301658, 0.4286297340409113, 0.7599723344051955, 0.9709954655858534, 0.5858166901036461, 0.7976237669509009, 0.2838829847359938, 0.5358073561746164, 0.9534960257111451, 0.866047188604062, 0.9186711084270732, 0.23849928311551094, 0.6737089030545828, 0.9716228067832702, 0.058124678542743835, 0.47186122092084537, 0.09051618632471803, 0.18498932305984406, 0.12714149395225105, 0.867386705616588, 0.8039557906378431, 0.4438447162909437, 0.03292863119137257, 0.4083196374971101, 0.6761261824275941, 0.47721054396058926, 0.03151700910036814, 0.4923681153555457, 0.6284200826822487, 0.3763756824057609, 0.917212044962352, 0.24446988940597913, 0.43418617011689864, 0.09105826385545324, 0.40956935119618354, 0.07265059868383206, 0.6393729464789593, 0.9095080389420704, 0.8378600544947684, 0.9599884845505204, 0.48223368974253167, 0.9407827580651115, 0.3363808177809786, 0.024041131534217386, 0.8390964079583833, 0.9400163522627636, 0.034924352262878045, 0.33940235510304806, 0.18534848543467475, 0.03580603754075151, 0.10405144184019743, 0.4934476361759681, 0.314639204685237, 0.16832358827884364, 0.8987714288751443, 0.5955991874486486, 0.35940671628887144, 0.19734977096387585, 0.16465058240183195, 0.11934280101053851, 0.685050484343792, 0.7050608024293961, 0.784522742103571, 0.7334664244176387, 0.06741499838126441, 0.2489688399437845, 0.23763856010694573, 0.27838141182331955, 0.8695448461672404, 0.547187835987318, 0.27310620483256143, 0.9366882453741473, 0.8879518339473409, 0.636677215036254, 0.8130713413458381, 0.9338342545939075, 0.29613773702405366, 0.657306775684372, 0.47066481567107443, 0.8705506291699947, 0.6551780117989051, 0.990832047307326, 0.5065171055602662, 0.5029816519267658, 0.5998370668834446, 0.550258374545736, 0.6169114950407212, 0.34004481776096285, 0.45471002509890324, 0.8943650078470086, 0.9789981901917617, 0.8557406511905119, 0.8020288340248788, 0.3416197060641377, 0.2923486086984194, 0.278781077502554, 0.7161891791246909, 0.4135284205590971, 0.409380486743371, 0.8355139854037481, 0.5173835739714673, 0.38671149103737945, 0.9660031234437596, 0.9106209006300167, 0.8753871046995989, 0.9231840581566352, 0.12125432418669979, 0.2505607856036176, 0.2882425160868405, 0.5432445486540161, 0.4105374820347819, 0.9377322262486563, 0.5867566465977914, 0.49500672319932226, 0.07842758749392498, 0.06369887679751252, 0.7545563500200172, 0.7591663832627792, 0.6476803743909982, 0.1659830360998571, 0.9776003721452584, 0.14228020332857783, 0.1013117374269531, 0.06865217006156088, 0.27920297080671697, 0.7494480472495295, 0.32032592558903783, 0.16016481561441076, 0.19170002054744006, 0.7051148298619444, 0.36295800467406436, 0.2721515682589073, 0.9454916382241941, 0.08896792297660183, 0.9807071022924964, 0.8702211047523876, 0.8224513412943082, 0.22376938790732914, 0.19532785060368918, 0.6263798329344097, 0.6753092846674776, 0.5042388068956346, 0.23986160159000802, 0.00690106347988606, 0.07514034356870436, 0.5983183842306461, 0.3411226775783537, 0.8444164073184014, 0.22439897357941097, 0.8246511280533838, 0.988960906831389, 0.03487353309843366, 0.7129270580258253, 0.9151147106267997, 0.3629003080714438, 0.1934566801091786, 0.5923120983924642, 0.44756921369870684, 0.8299484740357722, 0.6655628030950935, 0.7911060230560409, 0.45358738514911223, 0.11627010831527052, 0.8441056045260752, 0.09875626556765282, 0.755726060811237, 0.6969941125953589, 0.05986673769548989, 0.5902976199708095, 0.8949940549933123, 0.7382894358855404, 0.2664155310260169, 0.8877114697974581, 0.05701613528526306, 0.09867866013690607, 0.8328845789798932, 0.9420242763941248, 0.37914356544856775, 0.18684284678924423, 0.0799105141365436, 0.8487372278343337, 0.10216486250909274, 0.40633246722518324, 0.6793928013373823, 0.2094244598802134, 0.7338002441143087, 0.18783291926823875, 0.746473561117049, 0.9940793233119941, 0.06434535338739233, 0.8210201800027194, 0.05650699600962328, 0.38784957012380483, 0.13463139782803069, 0.8819615754794897, 0.2515360367670798, 0.08087732947561832, 0.8140325646988358, 0.782906423214805, 0.8947624551675152, 0.9111090912163098, 0.7840911491492375, 0.5723157942962758, 0.28770591595908857, 0.9217547146680796, 0.1647954137318446, 0.10646075702527247, 0.13662285474189795, 0.3616877554844493, 0.25168330960878305, 0.429349400826042, 0.031658234812691255, 0.2996737909291026, 0.48896454531373823, 0.9684101143415146, 0.8627245528212449, 0.025896310891397323, 0.303677051663801, 0.1790006747880054, 0.5102903572797703, 0.750911713292444, 0.38662343845764435, 0.18353610286818678, 0.126963719191451, 0.8447926671958864, 0.5685091854818741, 0.6550739941292127, 0.40026107569862024, 0.9657730236758066, 0.8506552049331029, 0.9846457603888858, 0.8351376938896239, 0.5148730756359207, 0.30037064753700493, 0.6683723046680513, 0.16923187358856728, 0.5313186658501128, 0.7801290011619055, 0.2598636127476923, 0.9955880003301975, 0.6465804480397815, 0.39265977472083335, 0.47414753533382215, 0.13765416982166745, 0.7613608714681567, 0.6329019320822552, 0.2511855284078234, 0.9716085142916913, 0.3531405134521378, 0.8264985237021314, 0.5620712202054944, 0.9892204887232992, 0.27876700914928476, 0.1329018595043162, 0.33121590955317115, 0.22980257438650087, 0.11444676160517353, 0.14303193569514228, 0.8319438935585682, 0.7620833735577972, 0.3241220600693051, 0.45283130079164347, 0.8864047421074285, 0.3550421908093988, 0.7529428416944839, 0.852944867843145, 0.8090034234488831, 0.015198676641192566, 0.19282332177856543, 0.9158726050630516, 0.2765825586162004, 0.6588012942321707, 0.8187586265551686, 0.03005997036902952, 0.9691849617386132, 0.27076309550770683, 0.6374011520639116, 0.9447957595506289, 0.3021748231272967, 0.870894263110552, 0.5543174650139618, 0.8532474518339176, 0.9372040127105596, 0.5651765667908824, 0.361768158499977, 0.01482297063468474, 0.45386974837369665, 0.7811769880286591, 0.1908565118768979, 0.3450496440655678, 0.0006924303592095171, 0.8326667590573007, 0.9873587971631453, 0.8467355305452563, 0.54173829214448, 0.08983990578423795, 0.739292458273785, 0.5788760285111699, 0.9129435644674967, 0.841127537173167, 0.8723868714397195, 0.9968217071676293, 0.9107641835811136, 0.4147830026772593, 0.7568766145128655, 0.6360328702769632, 0.61037490037664, 0.6556225837648071, 0.5703717841716494, 0.5744159416744337, 0.37437865481879384, 0.6602294107043496, 0.612737100639885, 0.9491792185268935, 0.6849113191231604, 0.899791205596821, 0.746664883743479, 0.347701575203092, 0.32889365902495804, 0.20125752125440388]
# b = np.random.rand(605)
# np.random.seed(0)
# b = np.random.rand(605)
# b = b.tolist()
cout = [0] * 5


def symbol_encode(atom):
    possible_atom = ['C', 'N', 'O', 'F', 'P', 'Cl', 'Br', 'I']
    # b= np.array(possible_atom)
    c = 0
    a = atom.GetSymbol()
    for i in range(8):
        if a == possible_atom[i]:
            c = i
            break
        else:
            c = 8
    return c


def hybrid_encode(atom):
    possible_hybrid = [Chem.rdchem.HybridizationType.SP,
                       Chem.rdchem.HybridizationType.SP2,
                       Chem.rdchem.HybridizationType.SP3,
                       Chem.rdchem.HybridizationType.SP3D]
    k = 0
    # b = np.array(possible_hybrid)
    a = atom.GetHybridization()
    for i in range(4):
        if a == possible_hybrid[i]:
            k = i
            break
    return k


def bond_encode(smi):
    m = Chem.MolFromSmiles(smi)
    # ri = m.GetRingInfo()
    nd2 = np.zeros((50, 10), int)
    bonds = m.GetBonds()
    ssr = Chem.GetSymmSSSR(m)
    b = len(ssr)
    k = 0
    possible_bond = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE]
    for bond in bonds:
        a = bond.GetBondType()
        for i in range(3):
            if a == possible_bond[i]:
                nd2[k, 0] = i + 1  # print('k=',k,end='\t')
                break
            if a != possible_bond[i] and i == 2:
                nd2[k, 0] = 0  # print('k=', k, end='\t')
        if bond.GetIsAromatic() == True:
            y = 1
            nd2[k, 1] = y
        else:
            nd2[k, 1] = 0
        # print('m=',m,end='\t ')
        if bond.GetIsConjugated() == True:
            nd2[k, 2] = 1
        else:
            nd2[k, 2] = 0
        # print('n=',n,end='\t')
        # print(b)
        if b <= 7:
            for i in range(b):
                d = (list(ssr[i]))
                ck = bond.GetBeginAtomIdx()
                ce = bond.GetEndAtomIdx()
                if ck and ce in d:
                    nd2[k, 3 + i] = 1
                else:
                    nd2[k, 3 + i] = 0
                # print('l=',l,end='\t')
        if b > 7:
            for i in range(7):
                d = (list(ssr[i]))
                ck = bond.GetBeginAtomIdx()
                ce = bond.GetEndAtomIdx()
                if ck and ce in d:
                    nd2[k, 3 + i] = 1
                else:
                    nd2[k, 3 + i] = 0
        k = k + 1
        if k == 5:
            break
    # print(nd2)
    return nd2


def encode_feature(smi):
    nd1 = np.zeros((60, 12), int)
    m = Chem.MolFromSmiles(smi)
    atoms = m.GetAtoms()
    ssr = Chem.GetSymmSSSR(m)
    # print('\t'.join(['symbol', 'degeree', 'charge', 'hybrid','jiadianzi','ChiralType','ring_information']))
    b = len(ssr)
    # print(b)
    k = 0
    for atom in atoms:
        # a = []
        c = atom.GetIdx()
        # a.append(symbol_encode(atom))
        # a.append(atom.GetDegree())
        # a.append(atom.GetFormalCharge())
        # a.append(hybrid_encode(atom))
        # a.append(atom.GetNumRadicalElectrons())
        # a.append(chiral_encode(atom))
        # print(atom.GetIdx())
        # print(atom.GetImplicitValence())
        nd1[k, 0] = symbol_encode(atom)
        nd1[k, 1] = atom.GetDegree()
        nd1[k, 2] = atom.GetFormalCharge()
        nd1[k, 3] = hybrid_encode(atom)
        nd1[k, 4] = atom.GetNumRadicalElectrons()
        nd1[k, 5] = chiral_encode(atom)
        if b <= 6:
            for i in range(b):
                if c in list(ssr[i]):
                    nd1[k, 6 + i] = 1
                else:
                    nd1[k, 6 + i] = 0
        if b > 6:
            for i in range(6):
                if c in list(ssr[i]):
                    nd1[k, 6 + i] = 1
                else:
                    nd1[k, 6 + i] = 0
        k = k + 1
        if k == 60:
            break
    # print(nd1)
    # print(k)
    return nd1


def chiral_encode(atom):
    possible_chiral = [Chem.rdchem.ChiralType.CHI_UNSPECIFIED, Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
                       Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW]
    b = atom.GetChiralTag()
    k = 0
    for i in range(3):
        if b == possible_chiral[i]:
            k = i
            break
    return k


def encode_anjisuan(str):
    listone = ['G', 'A', 'V', 'L', 'I', 'P', 'F', 'Y', 'W', 'S', 'T', 'C', 'M', 'N', 'Q', 'D', 'E', 'K', 'R', 'H']
    listtwo = [0] * 1400
    # print(str)
    num = len(str)
    # print(num)
    if num <= 1400:
        for i in range(num):
            for j in range(20):
                if str[i] == listone[j]:
                    listtwo[i] = (j + 1) / 20
                    break
    if num > 1400:
        for i in range(1400):
            for j in range(20):
                if str[i] == listone[j]:
                    listtwo[i] = (j + 1) / 20
                    break
            # if str[i] != listone[j] and j == 19:
            #     listtwo.append(0)
    # print(len(listtwo))
    return listtwo


data_excel = xlrd.open_workbook('/home/tisen/YXH/qcnn代码——药物靶点识别/ywbd.xls')
names = data_excel.sheet_names()
# print(names)
table = data_excel.sheets()[0]
smiles_valus = table.col_values(5, 1)
anjisuan_values = table.col_values(4, 1)
affinity_truth = table.col_values(2, 1)
# smilesY = encode_feature(smiles_valus[0])
# smilesB = bond_encode(smiles_valus[0])
# anjisuan = encode_anjisuan(anjisuan_values[0])
try:
    simulator_gpu = Aer.get_backend('aer_simulator')
    simulator_gpu.set_options(device='GPU')
except AerError as e:
    print(e)
# 11组线路
qr11 = QuantumRegister(5)
cr11 = ClassicalRegister(1)
qc11 = QuantumCircuit(qr11, cr11)
theta11 = []
for i in range(4):
    theta11_str = str(1) + str(1) + 'Θ' + str(i)
    theta11.append(Parameter(theta11_str))
# params = ParameterVector("θ",length=4)
for i in range(4):
    qc11.ry(np.pi / 2 * theta11[i], qr11[i + 1])
for i in range(3):
    qc11.cnot(qr11[i + 1], qr11[i + 2])
qc11.barrier()
phi11 = []
for j in range(5):
    for i in range(4):
        phi11_str = str(1) + str(1) + "e" + str(j) + "_" + "a" + str(i)
        phi11.append(Parameter(phi11_str))
for j in range(5):
    for i in range(4):
        qc11.ry(phi11[4 * j + i], qr11[i + 1])
    for k in range(3):
        qc11.cnot(qr11[k + 1], qr11[k + 2])
    qc11.barrier()
qc11.h(qr11[0])
for i in range(4):
    qc11.cz(qr11[0], qr11[i + 1])
qc11.h(qr11[0])
qc11.barrier()
qc11.measure(qr11[0], cr11)
# 12组线路
qr12 = QuantumRegister(5)
cr12 = ClassicalRegister(1)
qc12 = QuantumCircuit(qr12, cr12)
theta12 = []
for i in range(4):
    theta12_str = str(1) + str(2) + 'Θ' + str(i)
    theta12.append(Parameter(theta12_str))
# params = ParameterVector("θ",length=4)
for i in range(4):
    qc12.ry(np.pi / 2 * theta12[i], qr12[i + 1])
for i in range(3):
    qc12.cnot(qr12[i + 1], qr12[i + 2])
qc12.barrier()
phi12 = []
for j in range(5):
    for i in range(4):
        phi12_str = str(1) + str(2) + "e" + str(j) + "_" + "a" + str(i)
        phi12.append(Parameter(phi12_str))
for j in range(5):
    for i in range(4):
        qc12.ry(phi12[4 * j + i], qr12[i + 1])
    for k in range(3):
        qc12.cnot(qr12[k + 1], qr12[k + 2])
    qc12.barrier()
qc12.h(qr12[0])
for i in range(4):
    qc12.cz(qr12[0], qr12[i + 1])
qc12.h(qr12[0])
qc12.barrier()
qc12.measure(qr12[0], cr12)
# qc12.draw('mpl')
# plt.show()

# 13组线路
qr13 = QuantumRegister(5)
cr13 = ClassicalRegister(1)
qc13 = QuantumCircuit(qr13, cr13)
theta13 = []
for i in range(4):
    theta13_str = str(1) + str(3) + 'Θ' + str(i)
    theta13.append(Parameter(theta13_str))
    # params = ParameterVector("θ",length=4)
for i in range(4):
    qc13.ry(np.pi / 2 * theta13[i], qr13[i + 1])
for i in range(3):
    qc13.cnot(qr13[i + 1], qr13[i + 2])
qc13.barrier()
phi13 = []
for j in range(5):
    for i in range(4):
        phi13_str = str(1) + str(3) + "e" + str(j) + "_" + "a" + str(i)
        phi13.append(Parameter(phi13_str))
for j in range(5):
    for i in range(4):
        qc13.ry(phi13[4 * j + i], qr13[i + 1])
    for k in range(3):
        qc13.cnot(qr13[k + 1], qr13[k + 2])
    qc13.barrier()
qc13.h(qr13[0])
for i in range(4):
    qc13.cz(qr13[0], qr13[i + 1])
qc13.h(qr13[0])
qc13.barrier()
qc13.measure(qr13[0], cr13)
# 14组线路
qr14 = QuantumRegister(5)
cr14 = ClassicalRegister(1)
qc14 = QuantumCircuit(qr14, cr14)
theta14 = []
for i in range(4):
    theta14_str = str(1) + str(4) + 'Θ' + str(i)
    theta14.append(Parameter(theta14_str))
    # params = ParameterVector("θ",length=4)
for i in range(4):
    qc14.ry(np.pi / 2 * theta14[i], qr14[i + 1])
for i in range(3):
    qc14.cnot(qr14[i + 1], qr14[i + 2])
qc14.barrier()
phi14 = []
for j in range(5):
    for i in range(4):
        phi14_str = str(1) + str(4) + "e" + str(j) + "_" + "a" + str(i)
        phi14.append(Parameter(phi14_str))
for j in range(5):
    for i in range(4):
        qc14.ry(phi14[4 * j + i], qr14[i + 1])
    for k in range(3):
        qc14.cnot(qr14[k + 1], qr14[k + 2])
    qc14.barrier()
qc14.h(qr14[0])
for i in range(4):
    qc14.cz(qr14[0], qr14[i + 1])
qc14.h(qr14[0])
qc14.barrier()
qc14.measure(qr14[0], cr14)
# 15组线路
qr15 = QuantumRegister(5)
cr15 = ClassicalRegister(1)
qc15 = QuantumCircuit(qr15, cr15)
theta15 = []
for i in range(4):
    theta15_str = str(1) + str(5) + 'Θ' + str(i)
    theta15.append(Parameter(theta15_str))
    # params = ParameterVector("θ",length=4)
for i in range(4):
    qc15.ry(np.pi / 2 * theta15[i], qr15[i + 1])
for i in range(3):
    qc15.cnot(qr15[i + 1], qr15[i + 2])
qc15.barrier()
phi15 = []
for j in range(5):
    for i in range(4):
        phi15_str = str(1) + str(5) + "e" + str(j) + "_" + "a" + str(i)
        phi15.append(Parameter(phi15_str))
for j in range(5):
    for i in range(4):
        qc15.ry(phi15[4 * j + i], qr15[i + 1])
    for k in range(3):
        qc15.cnot(qr15[k + 1], qr15[k + 2])
    qc15.barrier()
qc15.h(qr15[0])
for i in range(4):
    qc15.cz(qr15[0], qr15[i + 1])
qc15.h(qr15[0])
qc15.barrier()
qc15.measure(qr15[0], cr15)
# 21组线路
qr21 = QuantumRegister(5)
cr21 = ClassicalRegister(1)
qc21 = QuantumCircuit(qr21, cr21)
theta21 = []
for i in range(4):
    theta21_str = str(2) + str(1) + 'Θ' + str(i)
    theta21.append(Parameter(theta21_str))
    # params = ParameterVector("θ",length=4)
for i in range(4):
    qc21.ry(np.pi / 2 * theta21[i], qr21[i + 1])
for i in range(3):
    qc21.cnot(qr21[i + 1], qr21[i + 2])
qc21.barrier()
phi21 = []
for j in range(5):
    for i in range(4):
        phi21_str = str(2) + str(1) + "e" + str(j) + "_" + "a" + str(i)
        phi21.append(Parameter(phi21_str))
for j in range(5):
    for i in range(4):
        qc21.ry(phi21[4 * j + i], qr21[i + 1])
    for k in range(3):
        qc21.cnot(qr21[k + 1], qr21[k + 2])
    qc21.barrier()
qc21.h(qr21[0])
for i in range(4):
    qc21.cz(qr21[0], qr21[i + 1])
qc21.h(qr21[0])
qc21.barrier()
qc21.measure(qr21[0], cr21)
# 22组线路
qr22 = QuantumRegister(5)
cr22 = ClassicalRegister(1)
qc22 = QuantumCircuit(qr22, cr22)
theta22 = []
for i in range(4):
    theta22_str = str(2) + str(2) + 'Θ' + str(i)
    theta22.append(Parameter(theta22_str))
    # params = ParameterVector("θ",length=4)
for i in range(4):
    qc22.ry(np.pi / 2 * theta22[i], qr22[i + 1])
for i in range(3):
    qc22.cnot(qr22[i + 1], qr22[i + 2])
qc22.barrier()
phi22 = []
for j in range(5):
    for i in range(4):
        phi22_str = str(2) + str(2) + "e" + str(j) + "_" + "a" + str(i)
        phi22.append(Parameter(phi22_str))
for j in range(5):
    for i in range(4):
        qc22.ry(phi22[4 * j + i], qr22[i + 1])
    for k in range(3):
        qc22.cnot(qr22[k + 1], qr22[k + 2])
    qc22.barrier()
qc22.h(qr22[0])
for i in range(4):
    qc22.cz(qr22[0], qr22[i + 1])
qc22.h(qr22[0])
qc22.barrier()
qc22.measure(qr22[0], cr22)
# 23组线路
qr23 = QuantumRegister(5)
cr23 = ClassicalRegister(1)
qc23 = QuantumCircuit(qr23, cr23)
theta23 = []
for i in range(4):
    theta23_str = str(2) + str(3) + 'Θ' + str(i)
    theta23.append(Parameter(theta23_str))
    # params = ParameterVector("θ",length=4)
for i in range(4):
    qc23.ry(np.pi / 2 * theta23[i], qr23[i + 1])
for i in range(3):
    qc23.cnot(qr23[i + 1], qr23[i + 2])
qc23.barrier()
phi23 = []
for j in range(5):
    for i in range(4):
        phi23_str = str(2) + str(3) + "e" + str(j) + "_" + "a" + str(i)
        phi23.append(Parameter(phi23_str))
for j in range(5):
    for i in range(4):
        qc23.ry(phi23[4 * j + i], qr23[i + 1])
    for k in range(3):
        qc23.cnot(qr23[k + 1], qr23[k + 2])
    qc23.barrier()
qc23.h(qr23[0])
for i in range(4):
    qc23.cz(qr23[0], qr23[i + 1])
qc23.h(qr23[0])
qc23.barrier()
qc23.measure(qr23[0], cr23)
# 24组线路
qr24 = QuantumRegister(5)
cr24 = ClassicalRegister(1)
qc24 = QuantumCircuit(qr24, cr24)
theta24 = []
for i in range(4):
    theta24_str = str(2) + str(4) + 'Θ' + str(i)
    theta24.append(Parameter(theta24_str))
    # params = ParameterVector("θ",length=4)
for i in range(4):
    qc24.ry(np.pi / 2 * theta24[i], qr24[i + 1])
for i in range(3):
    qc24.cnot(qr24[i + 1], qr24[i + 2])
qc24.barrier()
phi24 = []
for j in range(5):
    for i in range(4):
        phi24_str = str(2) + str(4) + "e" + str(j) + "_" + "a" + str(i)
        phi24.append(Parameter(phi24_str))
for j in range(5):
    for i in range(4):
        qc24.ry(phi24[4 * j + i], qr24[i + 1])
    for k in range(3):
        qc24.cnot(qr24[k + 1], qr24[k + 2])
    qc24.barrier()
qc24.h(qr24[0])
for i in range(4):
    qc24.cz(qr24[0], qr24[i + 1])
qc24.h(qr24[0])
qc24.barrier()
qc24.measure(qr24[0], cr24)
# 25组线路
qr25 = QuantumRegister(5)
cr25 = ClassicalRegister(1)
qc25 = QuantumCircuit(qr25, cr25)
theta25 = []
for i in range(4):
    theta25_str = str(2) + str(5) + 'Θ' + str(i)
    theta25.append(Parameter(theta25_str))
    # params = ParameterVector("θ",length=4)
for i in range(4):
    qc25.ry(np.pi / 2 * theta25[i], qr25[i + 1])
for i in range(3):
    qc25.cnot(qr25[i + 1], qr25[i + 2])
qc25.barrier()
phi25 = []
for j in range(5):
    for i in range(4):
        phi25_str = str(2) + str(5) + "e" + str(j) + "_" + "a" + str(i)
        phi25.append(Parameter(phi25_str))
for j in range(5):
    for i in range(4):
        qc25.ry(phi25[4 * j + i], qr25[i + 1])
    for k in range(3):
        qc25.cnot(qr25[k + 1], qr25[k + 2])
    qc25.barrier()
qc25.h(qr25[0])
for i in range(4):
    qc25.cz(qr25[0], qr25[i + 1])
qc25.h(qr25[0])
qc25.barrier()
qc25.measure(qr25[0], cr25)
# 31组线路
qr31 = QuantumRegister(5)
cr31 = ClassicalRegister(1)
qc31 = QuantumCircuit(qr31, cr31)
theta31 = []
for i in range(4):
    theta31_str = str(3) + str(1) + 'Θ' + str(i)
    theta31.append(Parameter(theta31_str))
    # params = ParameterVector("θ",length=4)
for i in range(4):
    qc31.ry(np.pi / 2 * theta31[i], qr31[i + 1])
for i in range(3):
    qc31.cnot(qr31[i + 1], qr31[i + 2])
qc31.barrier()
phi31 = []
for j in range(5):
    for i in range(4):
        phi31_str = str(3) + str(1) + "e" + str(j) + "_" + "a" + str(i)
        phi31.append(Parameter(phi31_str))
for j in range(5):
    for i in range(4):
        qc31.ry(phi31[4 * j + i], qr31[i + 1])
    for k in range(3):
        qc31.cnot(qr31[k + 1], qr31[k + 2])
    qc31.barrier()
qc31.h(qr31[0])
for i in range(4):
    qc31.cz(qr31[0], qr31[i + 1])
qc31.h(qr31[0])
qc31.barrier()
qc31.measure(qr31[0], cr31)
# 32组线路
qr32 = QuantumRegister(5)
cr32 = ClassicalRegister(1)
qc32 = QuantumCircuit(qr32, cr32)
theta32 = []
for i in range(4):
    theta32_str = str(3) + str(2) + 'Θ' + str(i)
    theta32.append(Parameter(theta32_str))
    # params = ParameterVector("θ",length=4)
for i in range(4):
    qc32.ry(np.pi / 2 * theta32[i], qr32[i + 1])
for i in range(3):
    qc32.cnot(qr32[i + 1], qr32[i + 2])
qc32.barrier()
phi32 = []
for j in range(5):
    for i in range(4):
        phi32_str = str(3) + str(2) + "e" + str(j) + "_" + "a" + str(i)
        phi32.append(Parameter(phi32_str))
for j in range(5):
    for i in range(4):
        qc32.ry(phi32[4 * j + i], qr32[i + 1])
    for k in range(3):
        qc32.cnot(qr32[k + 1], qr32[k + 2])
    qc32.barrier()
qc32.h(qr32[0])
for i in range(4):
    qc32.cz(qr32[0], qr32[i + 1])
qc32.h(qr32[0])
qc32.barrier()
qc32.measure(qr32[0], cr32)
# 33组线路
qr33 = QuantumRegister(5)
cr33 = ClassicalRegister(1)
qc33 = QuantumCircuit(qr33, cr33)
theta33 = []
for i in range(4):
    theta33_str = str(3) + str(3) + 'Θ' + str(i)
    theta33.append(Parameter(theta33_str))
    # params = ParameterVector("θ",length=4)
for i in range(4):
    qc33.ry(np.pi / 2 * theta33[i], qr33[i + 1])
for i in range(3):
    qc33.cnot(qr33[i + 1], qr33[i + 2])
qc33.barrier()
phi33 = []
for j in range(5):
    for i in range(4):
        phi33_str = str(3) + str(3) + "e" + str(j) + "_" + "a" + str(i)
        phi33.append(Parameter(phi33_str))
for j in range(5):
    for i in range(4):
        qc33.ry(phi33[4 * j + i], qr33[i + 1])
    for k in range(3):
        qc33.cnot(qr33[k + 1], qr33[k + 2])
    qc33.barrier()
qc33.h(qr33[0])
for i in range(4):
    qc33.cz(qr33[0], qr33[i + 1])
qc33.h(qr33[0])
qc33.barrier()
qc33.measure(qr33[0], cr33)
# 34组线路
qr34 = QuantumRegister(5)
cr34 = ClassicalRegister(1)
qc34 = QuantumCircuit(qr34, cr34)
theta34 = []
for i in range(4):
    theta34_str = str(3) + str(4) + 'Θ' + str(i)
    theta34.append(Parameter(theta34_str))
    # params = ParameterVector("θ",length=4)
for i in range(4):
    qc34.ry(np.pi / 2 * theta34[i], qr34[i + 1])
for i in range(3):
    qc34.cnot(qr34[i + 1], qr34[i + 2])
qc34.barrier()
phi34 = []
for j in range(5):
    for i in range(4):
        phi34_str = str(3) + str(4) + "e" + str(j) + "_" + "a" + str(i)
        phi34.append(Parameter(phi34_str))
for j in range(5):
    for i in range(4):
        qc34.ry(phi34[4 * j + i], qr34[i + 1])
    for k in range(3):
        qc34.cnot(qr34[k + 1], qr34[k + 2])
    qc34.barrier()
qc34.h(qr34[0])
for i in range(4):
    qc34.cz(qr34[0], qr34[i + 1])
qc34.h(qr34[0])
qc34.barrier()
qc34.measure(qr34[0], cr34)
# 35组线路
qr35 = QuantumRegister(5)
cr35 = ClassicalRegister(1)
qc35 = QuantumCircuit(qr35, cr35)
theta35 = []
for i in range(4):
    theta35_str = str(3) + str(5) + 'Θ' + str(i)
    theta35.append(Parameter(theta35_str))
    # params = ParameterVector("θ",length=4)
for i in range(4):
    qc35.ry(np.pi / 2 * theta35[i], qr35[i + 1])
for i in range(3):
    qc35.cnot(qr35[i + 1], qr35[i + 2])
qc35.barrier()
phi35 = []
for j in range(5):
    for i in range(4):
        phi35_str = str(3) + str(5) + "e" + str(j) + "_" + "a" + str(i)
        phi35.append(Parameter(phi35_str))
for j in range(5):
    for i in range(4):
        qc35.ry(phi35[4 * j + i], qr35[i + 1])
    for k in range(3):
        qc35.cnot(qr35[k + 1], qr35[k + 2])
    qc35.barrier()
qc35.h(qr35[0])
for i in range(4):
    qc35.cz(qr35[0], qr35[i + 1])
qc35.h(qr35[0])
qc35.barrier()
qc35.measure(qr35[0], cr35)
# 41组线路
qr41 = QuantumRegister(5)
cr41 = ClassicalRegister(1)
qc41 = QuantumCircuit(qr41, cr41)
theta41 = []
for i in range(4):
    theta41_str = str(4) + str(1) + 'Θ' + str(i)
    theta41.append(Parameter(theta41_str))
    # params = ParameterVector("θ",length=4)
for i in range(4):
    qc41.ry(np.pi / 2 * theta41[i], qr41[i + 1])
for i in range(3):
    qc41.cnot(qr41[i + 1], qr41[i + 2])
qc41.barrier()
phi41 = []
for j in range(5):
    for i in range(4):
        phi41_str = str(4) + str(1) + "e" + str(j) + "_" + "a" + str(i)
        phi41.append(Parameter(phi41_str))
for j in range(5):
    for i in range(4):
        qc41.ry(phi41[4 * j + i], qr41[i + 1])
    for k in range(3):
        qc41.cnot(qr41[k + 1], qr41[k + 2])
    qc41.barrier()
qc41.h(qr41[0])
for i in range(4):
    qc41.cz(qr41[0], qr41[i + 1])
qc41.h(qr41[0])
qc41.barrier()
qc41.measure(qr41[0], cr41)
# 42组线路
qr42 = QuantumRegister(5)
cr42 = ClassicalRegister(1)
qc42 = QuantumCircuit(qr42, cr42)
theta42 = []
for i in range(4):
    theta42_str = str(4) + str(2) + 'Θ' + str(i)
    theta42.append(Parameter(theta42_str))
    # params = ParameterVector("θ",length=4)
for i in range(4):
    qc42.ry(np.pi / 2 * theta42[i], qr42[i + 1])
for i in range(3):
    qc42.cnot(qr42[i + 1], qr42[i + 2])
qc42.barrier()
phi42 = []
for j in range(5):
    for i in range(4):
        phi42_str = str(4) + str(2) + "e" + str(j) + "_" + "a" + str(i)
        phi42.append(Parameter(phi42_str))
for j in range(5):
    for i in range(4):
        qc42.ry(phi42[4 * j + i], qr42[i + 1])
    for k in range(3):
        qc42.cnot(qr42[k + 1], qr42[k + 2])
    qc42.barrier()
qc42.h(qr42[0])
for i in range(4):
    qc42.cz(qr42[0], qr42[i + 1])
qc42.h(qr42[0])
qc42.barrier()
qc42.measure(qr42[0], cr42)
# 43组线路
qr43 = QuantumRegister(5)
cr43 = ClassicalRegister(1)
qc43 = QuantumCircuit(qr43, cr43)
theta43 = []
for i in range(4):
    theta43_str = str(4) + str(3) + 'Θ' + str(i)
    theta43.append(Parameter(theta43_str))
    # params = ParameterVector("θ",length=4)
for i in range(4):
    qc43.ry(np.pi / 2 * theta43[i], qr43[i + 1])
for i in range(3):
    qc43.cnot(qr43[i + 1], qr43[i + 2])
qc43.barrier()
phi43 = []
for j in range(5):
    for i in range(4):
        phi43_str = str(4) + str(3) + "e" + str(j) + "_" + "a" + str(i)
        phi43.append(Parameter(phi43_str))
for j in range(5):
    for i in range(4):
        qc43.ry(phi43[4 * j + i], qr43[i + 1])
    for k in range(3):
        qc43.cnot(qr43[k + 1], qr43[k + 2])
    qc43.barrier()
qc43.h(qr43[0])
for i in range(4):
    qc43.cz(qr43[0], qr43[i + 1])
qc43.h(qr43[0])
qc43.barrier()
qc43.measure(qr43[0], cr43)
# 44组线路
qr44 = QuantumRegister(5)
cr44 = ClassicalRegister(1)
qc44 = QuantumCircuit(qr44, cr44)
theta44 = []
for i in range(4):
    theta44_str = str(4) + str(4) + 'Θ' + str(i)
    theta44.append(Parameter(theta44_str))
    # params = ParameterVector("θ",length=4)
for i in range(4):
    qc44.ry(np.pi / 2 * theta44[i], qr44[i + 1])
for i in range(3):
    qc44.cnot(qr44[i + 1], qr44[i + 2])
qc44.barrier()
phi44 = []
for j in range(5):
    for i in range(4):
        phi44_str = str(4) + str(4) + "e" + str(j) + "_" + "a" + str(i)
        phi44.append(Parameter(phi44_str))
for j in range(5):
    for i in range(4):
        qc44.ry(phi44[4 * j + i], qr44[i + 1])
    for k in range(3):
        qc44.cnot(qr44[k + 1], qr44[k + 2])
    qc44.barrier()
qc44.h(qr44[0])
for i in range(4):
    qc44.cz(qr44[0], qr44[i + 1])
qc44.h(qr44[0])
qc44.barrier()
qc44.measure(qr44[0], cr44)
# 45组线路
qr45 = QuantumRegister(5)
cr45 = ClassicalRegister(1)
qc45 = QuantumCircuit(qr45, cr45)
theta45 = []
for i in range(4):
    theta45_str = str(4) + str(5) + 'Θ' + str(i)
    theta45.append(Parameter(theta45_str))
    # params = ParameterVector("θ",length=4)
for i in range(4):
    qc45.ry(np.pi / 2 * theta45[i], qr45[i + 1])
for i in range(3):
    qc45.cnot(qr45[i + 1], qr45[i + 2])
qc45.barrier()
phi45 = []
for j in range(5):
    for i in range(4):
        phi45_str = str(4) + str(5) + "e" + str(j) + "_" + "a" + str(i)
        phi45.append(Parameter(phi45_str))
for j in range(5):
    for i in range(4):
        qc45.ry(phi45[4 * j + i], qr45[i + 1])
    for k in range(3):
        qc45.cnot(qr45[k + 1], qr45[k + 2])
    qc45.barrier()
qc45.h(qr45[0])
for i in range(4):
    qc45.cz(qr45[0], qr45[i + 1])
qc45.h(qr45[0])
qc45.barrier()
qc45.measure(qr45[0], cr45)
# 51组线路
qr51 = QuantumRegister(5)
cr51 = ClassicalRegister(1)
qc51 = QuantumCircuit(qr51, cr51)
theta51 = []
for i in range(4):
    theta51_str = str(5) + str(1) + 'Θ' + str(i)
    theta51.append(Parameter(theta51_str))
    # params = ParameterVector("θ",length=4)
for i in range(4):
    qc51.ry(np.pi / 2 * theta51[i], qr51[i + 1])
for i in range(3):
    qc51.cnot(qr51[i + 1], qr51[i + 2])
qc51.barrier()
phi51 = []
for j in range(5):
    for i in range(4):
        phi51_str = str(5) + str(1) + "e" + str(j) + "_" + "a" + str(i)
        phi51.append(Parameter(phi51_str))
for j in range(5):
    for i in range(4):
        qc51.ry(phi51[4 * j + i], qr51[i + 1])
    for k in range(3):
        qc51.cnot(qr51[k + 1], qr51[k + 2])
    qc51.barrier()
qc51.h(qr51[0])
for i in range(4):
    qc51.cz(qr51[0], qr51[i + 1])
qc51.h(qr51[0])
qc51.barrier()
qc51.measure(qr51[0], cr51)
# 52组线路
qr52 = QuantumRegister(5)
cr52 = ClassicalRegister(1)
qc52 = QuantumCircuit(qr52, cr52)
theta52 = []
for i in range(4):
    theta52_str = str(5) + str(2) + 'Θ' + str(i)
    theta52.append(Parameter(theta52_str))
    # params = ParameterVector("θ",length=4)
for i in range(4):
    qc52.ry(np.pi / 2 * theta52[i], qr52[i + 1])
for i in range(3):
    qc52.cnot(qr52[i + 1], qr52[i + 2])
qc52.barrier()
phi52 = []
for j in range(5):
    for i in range(4):
        phi52_str = str(5) + str(2) + "e" + str(j) + "_" + "a" + str(i)
        phi52.append(Parameter(phi52_str))
for j in range(5):
    for i in range(4):
        qc52.ry(phi52[4 * j + i], qr52[i + 1])
    for k in range(3):
        qc52.cnot(qr52[k + 1], qr52[k + 2])
    qc52.barrier()
qc52.h(qr52[0])
for i in range(4):
    qc52.cz(qr52[0], qr52[i + 1])
qc52.h(qr52[0])
qc52.barrier()
qc52.measure(qr52[0], cr52)
# 53组线路
qr53 = QuantumRegister(5)
cr53 = ClassicalRegister(1)
qc53 = QuantumCircuit(qr53, cr53)
theta53 = []
for i in range(4):
    theta53_str = str(5) + str(3) + 'Θ' + str(i)
    theta53.append(Parameter(theta53_str))
    # params = ParameterVector("θ",length=4)
for i in range(4):
    qc53.ry(np.pi / 2 * theta53[i], qr53[i + 1])
for i in range(3):
    qc53.cnot(qr53[i + 1], qr53[i + 2])
qc53.barrier()
phi53 = []
for j in range(5):
    for i in range(4):
        phi53_str = str(5) + str(3) + "e" + str(j) + "_" + "a" + str(i)
        phi53.append(Parameter(phi53_str))
for j in range(5):
    for i in range(4):
        qc53.ry(phi53[4 * j + i], qr53[i + 1])
    for k in range(3):
        qc53.cnot(qr53[k + 1], qr53[k + 2])
    qc53.barrier()
qc53.h(qr53[0])
for i in range(4):
    qc53.cz(qr53[0], qr53[i + 1])
qc53.h(qr53[0])
qc53.barrier()
qc53.measure(qr53[0], cr53)
# 54组线路
qr54 = QuantumRegister(5)
cr54 = ClassicalRegister(1)
qc54 = QuantumCircuit(qr54, cr54)
theta54 = []
for i in range(4):
    theta54_str = str(5) + str(4) + 'Θ' + str(i)
    theta54.append(Parameter(theta54_str))
    # params = ParameterVector("θ",length=4)
for i in range(4):
    qc54.ry(np.pi / 2 * theta54[i], qr54[i + 1])
for i in range(3):
    qc54.cnot(qr54[i + 1], qr54[i + 2])
qc54.barrier()
phi54 = []
for j in range(5):
    for i in range(4):
        phi54_str = str(5) + str(4) + "e" + str(j) + "_" + "a" + str(i)
        phi54.append(Parameter(phi54_str))
for j in range(5):
    for i in range(4):
        qc54.ry(phi54[4 * j + i], qr54[i + 1])
    for k in range(3):
        qc54.cnot(qr54[k + 1], qr54[k + 2])
    qc54.barrier()
qc54.h(qr54[0])
for i in range(4):
    qc54.cz(qr54[0], qr54[i + 1])
qc54.h(qr54[0])
qc54.barrier()
qc54.measure(qr54[0], cr54)
# 55组线路
qr55 = QuantumRegister(7)
cr55 = ClassicalRegister(1)
qc55 = QuantumCircuit(qr55, cr55)
theta55 = []
for i in range(6):
    theta55_str = str(5) + str(5) + 'Θ' + str(i)
    theta55.append(Parameter(theta55_str))
    # params = ParameterVector("θ",length=4)
for i in range(6):
    qc55.ry(np.pi / 2 * theta55[i], qr55[i + 1])
for i in range(5):
    qc55.cnot(qr55[i + 1], qr55[i + 2])
qc55.barrier()
phi55 = []
for j in range(5):
    for i in range(6):
        phi55_str = str(5) + str(5) + "e" + str(j) + "_" + "a" + str(i)
        phi55.append(Parameter(phi55_str))
for j in range(5):
    for i in range(6):
        qc55.ry(phi55[6 * j + i], qr55[i + 1])
    for k in range(5):
        qc55.cnot(qr55[k + 1], qr55[k + 2])
    qc55.barrier()
qc55.h(qr55[0])
for i in range(6):
    qc55.cz(qr55[0], qr55[i + 1])
qc55.h(qr55[0])
qc55.barrier()
qc55.measure(qr55[0], cr55)
# qc55.draw('mpl')
# plt.show()
# 全连接线路
qr6 = QuantumRegister(6)
cr6 = ClassicalRegister(1)
qc6 = QuantumCircuit(qr6, cr6)
theta6 = []
for i in range(5):
    theta6_str = str(6) + 'Θ' + str(i)
    theta6.append(Parameter(theta6_str))
for i in range(5):
    qc6.ry(np.pi / 2 * theta6[i], qr6[i + 1])
for i in range(4):
    qc6.cnot(qr6[i + 1], qr6[i + 2])
qc6.barrier()
phi1 = []
phi2 = []
phi3 = []
for i in range(5):
    phi1_str = "x" + "p" + str(i)
    phi1.append(Parameter(phi1_str))
for i in range(10):
    for j in range(4):
        phi3_str = "z" + str(i) + "_" + str(j) + str(j + 1)
        phi3.append(Parameter(phi3_str))
for i in range(10):
    for j in range(5):
        phi2_str = "y" + str(i) + "_0" + str(j)
        phi2.append(Parameter(phi2_str))
# 建立参数门
for i in range(5):
    qc6.rx(phi1[i], qr6[i + 1])
for i in range(4):
    qc6.rzz(phi3[i], qr6[i + 1], qr6[i + 2])
qc6.barrier()
for i in range(9):
    for j in range(5):
        qc6.ry(phi2[5 * i + j], qr6[j + 1])
        qc6.rx(phi1[j], qr6[j + 1])
    for k in range(4):
        qc6.rzz(phi3[(i + 1) * 4 + k], qr6[k + 1], qr6[k + 2])
    qc6.barrier()
for i in range(5):
    qc6.ry(phi2[45 + i], qr6[i + 1])
    qc6.rx(phi1[i], qr6[i + 1])
qc6.barrier()
qc6.h(qr6[0])
for i in range(5):
    qc6.cz(qr6[0], qr6[i + 1])
qc6.h(qr6[0])
qc6.measure(qr6[0], cr6)


def conv1(smilesY, p11, p12, p13, p14, p15):
    out11 = np.zeros((30, 6))
    # backend = BasicAer.get_backend('qasm_simulator')
    backend = simulator_gpu
    for j in range(0, 12, 2):
        for i in range(0, 60, 2):
            input = [smilesY[i, j], smilesY[i, j + 1], smilesY[i + 1, j], smilesY[i + 1, j + 1]]
            # print(input)
            paramter11_1 = {zip(theta11, input)}
            paramter11_2 = {zip(phi11, p11)}
            # backend = BasicAer.get_backend('qasm_simulator')

            job = execute(qc11, backend, parameter_binds=[{**paramter11_1, **paramter11_2}], shots=2000)
            result = job.result().get_counts()
            result0 = result['0'] / 2000
            out11[i // 2, j // 2] = result0
    # print(out11)
    out12 = np.zeros((16, 4))
    for j in range(0, 6, 2):
        for i in range(0, 30, 2):
            input = [out11[i, j], out11[i, j + 1], out11[i + 1, j], out11[i + 1, j + 1]]
            # print(input)
            paramter12_1 = {zip(theta12, input)}
            paramter12_2 = {zip(phi12, p12)}
            # backend = BasicAer.get_backend('qasm_simulator')
            job = execute(qc12, backend, parameter_binds=[{**paramter12_1, **paramter12_2}], shots=2000)
            result = job.result().get_counts()
            result0 = result['0'] / 2000
            out12[i // 2, j // 2] = result0
    # print(out12)
    out13 = np.zeros((8, 2))
    for j in range(0, 4, 2):
        for i in range(0, 16, 2):
            input = [out12[i, j], out12[i, j + 1], out12[i + 1, j], out12[i + 1, j + 1]]
            # print(input)
            paramter13_1 = {zip(theta13, input)}
            paramter13_2 = {zip(phi13, p13)}
            # backend = BasicAer.get_backend('qasm_simulator')
            job = execute(qc13, backend, parameter_binds=[{**paramter13_1, **paramter13_2}], shots=2000)
            result = job.result().get_counts()
            result0 = result['0'] / 2000
            out13[i // 2, j // 2] = result0
    # print(out13)
    out14 = np.zeros((4, 1))
    for j in range(0, 2, 2):
        for i in range(0, 8, 2):
            input = [out13[i, j], out13[i, j + 1], out13[i + 1, j], out13[i + 1, j + 1]]
            # print(input)
            paramter14_1 = {zip(theta14, input)}
            paramter14_2 = {zip(phi14, p14)}
            # backend = BasicAer.get_backend('qasm_simulator')
            job = execute(qc14, backend, parameter_binds=[{**paramter14_1, **paramter14_2}], shots=2000)
            result = job.result().get_counts()
            result0 = result['0'] / 2000
            out14[i // 2, j // 2] = result0
    # print(out14)
    input = [out14[0, 0], out14[1, 0], out14[2, 0], out14[3, 0]]
    paramter15_1 = {zip(theta15, input)}
    paramter15_2 = {zip(phi15, p15)}
    # backend = BasicAer.get_backend('qasm_simulator')
    job = execute(qc15, backend, parameter_binds=[{**paramter15_1, **paramter15_2}], shots=2000)
    result = job.result().get_counts()
    cout[0] = result['0'] / 2000
    # print(out15)
    return cout[0]


def conv2(smilesY, p21, p22, p23, p24, p25):
    out11 = np.zeros((30, 6))
    backend = simulator_gpu
    # backend = BasicAer.get_backend('qasm_simulator')
    for j in range(0, 12, 2):
        for i in range(0, 60, 2):
            input = [smilesY[i, j], smilesY[i, j + 1], smilesY[i + 1, j], smilesY[i + 1, j + 1]]
            # print(input)
            paramter11_1 = {zip(theta21, input)}
            paramter11_2 = {zip(phi21, p21)}
            # backend = BasicAer.get_backend('qasm_simulator')

            job = execute(qc21, backend, parameter_binds=[{**paramter11_1, **paramter11_2}], shots=2000)
            result = job.result().get_counts()
            result0 = result['0'] / 2000
            out11[i // 2, j // 2] = result0
    # print(out11)
    out12 = np.zeros((16, 4))
    for j in range(0, 6, 2):
        for i in range(0, 30, 2):
            input = [out11[i, j], out11[i, j + 1], out11[i + 1, j], out11[i + 1, j + 1]]
            # print(input)
            paramter12_1 = {zip(theta22, input)}
            paramter12_2 = {zip(phi22, p22)}
            # backend = BasicAer.get_backend('qasm_simulator')
            job = execute(qc22, backend, parameter_binds=[{**paramter12_1, **paramter12_2}], shots=2000)
            result = job.result().get_counts()
            result0 = result['0'] / 2000
            out12[i // 2, j // 2] = result0
    # print(out12)
    out13 = np.zeros((8, 2))
    for j in range(0, 4, 2):
        for i in range(0, 16, 2):
            input = [out12[i, j], out12[i, j + 1], out12[i + 1, j], out12[i + 1, j + 1]]
            # print(input)
            paramter13_1 = {zip(theta23, input)}
            paramter13_2 = {zip(phi23, p23)}
            # backend = BasicAer.get_backend('qasm_simulator')
            job = execute(qc23, backend, parameter_binds=[{**paramter13_1, **paramter13_2}], shots=2000)
            result = job.result().get_counts()
            result0 = result['0'] / 2000
            out13[i // 2, j // 2] = result0
    # print(out13)
    out14 = np.zeros((4, 1))
    for j in range(0, 2, 2):
        for i in range(0, 8, 2):
            input = [out13[i, j], out13[i, j + 1], out13[i + 1, j], out13[i + 1, j + 1]]
            # print(input)
            paramter14_1 = {zip(theta24, input)}
            paramter14_2 = {zip(phi24, p24)}
            # backend = BasicAer.get_backend('qasm_simulator')
            job = execute(qc24, backend, parameter_binds=[{**paramter14_1, **paramter14_2}], shots=2000)
            result = job.result().get_counts()
            result0 = result['0'] / 2000
            out14[i // 2, j // 2] = result0
    # print(out14)
    input = [out14[0, 0], out14[1, 0], out14[2, 0], out14[3, 0]]
    paramter15_1 = {zip(theta25, input)}
    paramter15_2 = {zip(phi25, p25)}
    # backend = BasicAer.get_backend('qasm_simulator')
    job = execute(qc25, backend, parameter_binds=[{**paramter15_1, **paramter15_2}], shots=2000)
    result = job.result().get_counts()
    cout[1] = result['0'] / 2000
    # print(out15)
    return cout[1]


def conv3(smilesB, p31, p32, p33, p34, p35):
    out11 = np.zeros((26, 6))
    backend = simulator_gpu
    # backend = BasicAer.get_backend('qasm_simulator')
    for j in range(0, 10, 2):
        for i in range(0, 50, 2):
            input = [smilesB[i, j], smilesB[i, j + 1], smilesB[i + 1, j], smilesB[i + 1, j + 1]]
            # print(input)
            paramter11_1 = {zip(theta31, input)}
            paramter11_2 = {zip(phi31, p31)}
            # backend = BasicAer.get_backend('qasm_simulator')

            job = execute(qc31, backend, parameter_binds=[{**paramter11_1, **paramter11_2}], shots=2000)
            result = job.result().get_counts()
            result0 = result['0'] / 2000
            out11[i // 2, j // 2] = result0
    # print(out11)
    out12 = np.zeros((14, 4))
    for j in range(0, 6, 2):
        for i in range(0, 26, 2):
            input = [out11[i, j], out11[i, j + 1], out11[i + 1, j], out11[i + 1, j + 1]]
            # print(input)
            paramter12_1 = {zip(theta32, input)}
            paramter12_2 = {zip(phi32, p32)}
            # backend = BasicAer.get_backend('qasm_simulator')
            job = execute(qc32, backend, parameter_binds=[{**paramter12_1, **paramter12_2}], shots=2000)
            result = job.result().get_counts()
            result0 = result['0'] / 2000
            out12[i // 2, j // 2] = result0
    # print(out12)
    out13 = np.zeros((8, 2))
    for j in range(0, 4, 2):
        for i in range(0, 14, 2):
            input = [out12[i, j], out12[i, j + 1], out12[i + 1, j], out12[i + 1, j + 1]]
            # print(input)
            paramter13_1 = {zip(theta33, input)}
            paramter13_2 = {zip(phi33, p33)}
            # backend = BasicAer.get_backend('qasm_simulator')
            job = execute(qc33, backend, parameter_binds=[{**paramter13_1, **paramter13_2}], shots=2000)
            result = job.result().get_counts()
            result0 = result['0'] / 2000
            out13[i // 2, j // 2] = result0
    # print(out13)
    out14 = np.zeros((4, 1))
    for i in range(0, 2, 2):
        for j in range(0, 8, 2):
            input = [out13[i, j], out13[i, j + 1], out13[i + 1, j], out13[i + 1, j + 1]]
            # print(input)
            paramter14_1 = {zip(theta34, input)}
            paramter14_2 = {zip(phi34, p34)}
            # backend = BasicAer.get_backend('qasm_simulator')
            job = execute(qc34, backend, parameter_binds=[{**paramter14_1, **paramter14_2}], shots=2000)
            result = job.result().get_counts()
            result0 = result['0'] / 2000
            out14[i // 2, j // 2] = result0
    # print(out14)
    input = [out14[0, 0], out14[1, 0], out14[2, 0], out14[3, 0]]
    paramter15_1 = {zip(theta35, input)}
    paramter15_2 = {zip(phi35, p35)}
    # backend = BasicAer.get_backend('qasm_simulator')
    job = execute(qc35, backend, parameter_binds=[{**paramter15_1, **paramter15_2}], shots=2000)
    result = job.result().get_counts()
    cout[2] = result['0'] / 2000
    # print(out15)
    return cout[2]


def conv4(smilesB, p41, p42, p43, p44, p45):
    out11 = np.zeros((26, 6))
    backend = simulator_gpu
    # backend = BasicAer.get_backend('qasm_simulator')
    for j in range(0, 10, 2):
        for i in range(0, 50, 2):
            start = time.time()
            input = [smilesB[i, j], smilesB[i, j + 1], smilesB[i + 1, j], smilesB[i + 1, j + 1]]
            # print(input)
            paramter11_1 = {zip(theta41, input)}
            paramter11_2 = {zip(phi41, p41)}
            # backend = BasicAer.get_backend('qasm_simulator')

            job = execute(qc41, backend, parameter_binds=[{**paramter11_1, **paramter11_2}], shots=2000)
            result = job.result().get_counts()
            result0 = result['0'] / 2000
            out11[i // 2, j // 2] = result0
            end = time.time()
            print(str(round(end - start, 3)) + 's')
    # print(out11)
    out12 = np.zeros((14, 4))
    for j in range(0, 6, 2):
        for i in range(0, 26, 2):
            input = [out11[i, j], out11[i, j + 1], out11[i + 1, j], out11[i + 1, j + 1]]
            # print(input)
            paramter12_1 = {zip(theta42, input)}
            paramter12_2 = {zip(phi42, p42)}
            # backend = BasicAer.get_backend('qasm_simulator')
            job = execute(qc42, backend, parameter_binds=[{**paramter12_1, **paramter12_2}], shots=2000)
            result = job.result().get_counts()
            result0 = result['0'] / 2000
            out12[i // 2, j // 2] = result0
    # print(out12)
    out13 = np.zeros((8, 2))
    for j in range(0, 4, 2):
        for i in range(0, 14, 2):
            input = [out12[i, j], out12[i, j + 1], out12[i + 1, j], out12[i + 1, j + 1]]
            # print(input)
            paramter13_1 = {zip(theta43, input)}
            paramter13_2 = {zip(phi43, p43)}
            # backend = BasicAer.get_backend('qasm_simulator')
            job = execute(qc43, backend, parameter_binds=[{**paramter13_1, **paramter13_2}], shots=2000)
            result = job.result().get_counts()
            result0 = result['0'] / 2000
            out13[i // 2, j // 2] = result0
    # print(out13)
    out14 = np.zeros((4, 1))
    for j in range(0, 2, 2):
        for i in range(0, 8, 2):
            input = [out13[i, j], out13[i, j + 1], out13[i + 1, j], out13[i + 1, j + 1]]
            # print(input)
            paramter14_1 = {zip(theta44, input)}
            paramter14_2 = {zip(phi44, p44)}
            # backend = BasicAer.get_backend('qasm_simulator')
            job = execute(qc44, backend, parameter_binds=[{**paramter14_1, **paramter14_2}], shots=2000)
            result = job.result().get_counts()
            result0 = result['0'] / 2000
            out14[i // 2, j // 2] = result0
    # print(out14)
    input = [out14[0, 0], out14[1, 0], out14[2, 0], out14[3, 0]]
    paramter15_1 = {zip(theta45, input)}
    paramter15_2 = {zip(phi45, p45)}
    # backend = BasicAer.get_backend('qasm_simulator')
    job = execute(qc45, backend, parameter_binds=[{**paramter15_1, **paramter15_2}], shots=2000)
    result = job.result().get_counts()
    cout[3] = result['0'] / 2000
    # print(out15)
    return cout[3]


# for i in range(5):
#     smilesY = encode_feature(smiles_valus[i])
#     conv1(p11,p12,p13,p14,p15,smilesY,qc11,qc12,qc13,qc14,qc15,theta11,theta12,theta13,theta14,theta15,phi11,phi12,phi13,phi14,phi15)
#     conv1(p21,p22,p23,p24,p25,smilesY,qc21,qc22,qc23,qc24,qc25,theta21,theta22,theta23,theta24,theta25,phi21,phi22,phi23,phi24,phi25)
# for i in range(5):
#     smilesB = bond_encode(smiles_valus[i])
#     conv2(p31,p32,p33,p34,p35,smilesB,qc31,qc32,qc33,qc34,qc35,theta31,theta32,theta33,theta34,theta35,phi31,phi32,phi33,phi34,phi35)
#     conv2(p41,p42,p43,p44,p45,smilesB,qc41,qc42,qc43,qc44,qc45,theta41,theta42,theta43,theta44,theta45,phi41,phi42,phi43,phi44,phi45)

def conv5(anjisuanvalues, p51, p52, p53, p54, p55):
    out1 = [0] * 352
    anjisuan = encode_anjisuan(anjisuanvalues)
    backend = simulator_gpu
    # backend = BasicAer.get_backend('qasm_simulator')
    for i in range(0, 1400, 4):
        start = time.time()
        input = [anjisuan[i], anjisuan[i + 1], anjisuan[i + 2], anjisuan[i + 3]]
        # print(input)
        paramter11_1 = dict(zip(theta51, input))
        paramter11_2 = dict(zip(phi51, p51))
        # backend = BasicAer.get_backend('qasm_simulator')
        job = execute(qc51, backend, parameter_binds=[{**paramter11_1, **paramter11_2}], shots=2000)
        result = job.result().get_counts()
        result0 = result['0'] / 2000
        out1[i // 4] = result0
        end = time.time()
        print(str(round(end - start, 3)) + 's')
    # print(out11)
    out2 = [0] * 88
    for i in range(0, 352, 4):
        input = [out1[i], out1[i + 1], out1[i + 2], out1[i + 3]]
        # print(input)
        paramter12_1 = dict(zip(theta52, input))
        paramter12_2 = dict(zip(phi52, p52))
        # backend = BasicAer.get_backend('qasm_simulator')
        job = execute(qc52, backend, parameter_binds=[{**paramter12_1, **paramter12_2}], shots=2000)
        result = job.result().get_counts()
        result0 = result['0'] / 2000
        out2[i // 4] = result0
    # print(out12)
    out3 = [0] * 24
    for i in range(0, 88, 4):
        input = [out2[i], out2[i + 1], out2[i + 2], out2[i + 3]]
        # print(input)
        paramter13_1 = dict(zip(theta53, input))
        paramter13_2 = dict(zip(phi53, p53))
        # backend = BasicAer.get_backend('qasm_simulator')
        job = execute(qc53, backend, parameter_binds=[{**paramter13_1, **paramter13_2}], shots=2000)
        result = job.result().get_counts()
        result0 = result['0'] / 2000
        out3[i // 4] = result0
    # print(out13)
    out4 = [0] * 6
    for i in range(0, 24, 4):
        input = [out3[i], out3[i + 1], out3[i + 2], out3[i + 3]]
        # print(input)
        paramter14_1 = dict(zip(theta54, input))
        paramter14_2 = dict(zip(phi54, p54))
        # backend = BasicAer.get_backend('qasm_simulator')
        job = execute(qc54, backend, parameter_binds=[{**paramter14_1, **paramter14_2}], shots=2000)
        result = job.result().get_counts()
        result0 = result['0'] / 2000
        out4[i // 4] = result0
    # print(out14)
    input = [out4[0], out4[1], out4[2], out4[3], out4[4], out4[5]]
    paramter15_1 = dict(zip(theta55, input))
    paramter15_2 = dict(zip(phi55, p55))
    # backend = BasicAer.get_backend('qasm_simulator')
    job = execute(qc55, backend, parameter_binds=[{**paramter15_1, **paramter15_2}], shots=2000)
    result = job.result().get_counts()
    cout[4] = result['0'] / 2000
    # print(out5)
    return cout[4]


def Qfu_conec_cirt(p61, p62, p63):
    parameter0 = dict(zip(theta6, cout))
    parameter1 = dict(zip(phi1, p61))
    parameter2 = dict(zip(phi2, p62))
    parameter3 = dict(zip(phi3, p63))
    parameter0.update(parameter1)
    parameter0.update(parameter2)
    parameter0.update(parameter3)
    parameter = parameter0
    #backend = simulator_gpu
    backend = BasicAer.get_backend('qasm_simulator')
    job = execute(qc6, backend, parameter_binds=[parameter0], shots=2000)
    result = job.result().get_counts()
    out = result['0'] / 2000
    # print(out)
    # plot_histogram(result)
    return out


def loss_function(para):
    loss_value = []
    p11 = para[0:20]
    p12 = para[20:40]
    p13 = para[40:60]
    p14 = para[60:80]
    p15 = para[80:100]
    p21 = para[100:120]
    p22 = para[120:140]
    p23 = para[140:160]
    p24 = para[160:180]
    p25 = para[180:200]
    p31 = para[200:220]
    p32 = para[220:240]
    p33 = para[240:260]
    p34 = para[260:280]
    p35 = para[280:300]
    p41 = para[300:320]
    p42 = para[320:340]
    p43 = para[340:360]
    p44 = para[360:380]
    p45 = para[380:400]
    p51 = para[400:420]
    p52 = para[420:440]
    p53 = para[440:460]
    p54 = para[460:480]
    p55 = para[480:510]
    p61 = para[510:515]
    p62 = para[515:565]
    p63 = para[565:605]
    # b_para = para[605]
    # cout = [0]*5
    for i in range(150):
        smilesY = encode_feature(smiles_valus[i])
        smilesB = bond_encode(smiles_valus[i])
        anjisuan = encode_anjisuan(anjisuan_values[i])
        start = time.time()
        # p1 = multiprocessing.Process(target=conv1, args=(smilesY, p11, p12, p13, p14, p15))
        # p2 = multiprocessing.Process(target=conv2, args=(smilesY, p21, p22, p23, p24, p25))
        # p3 = multiprocessing.Process(target=conv3, args=(smilesB, p31, p32, p33, p34, p35))
        # p4 = multiprocessing.Process(target=conv4, args=(smilesB, p41, p42, p43, p44, p45))
        # p5 = multiprocessing.Process(target=conv5, args=(anjisuan, p51, p52, p53, p54, p55))
        # 创建线程池 最大并行进程数5个
        pool = multiprocessing.Pool(processes=5)
        # pool.apply_async 加入进程池，异步调用（无需等待函数执行完返回）第一个参数函数名 第二个参数 函数参数元组 .get() 获取函数调用的返回值
        pool.apply_async(conv1, (smilesY, p11, p12, p13, p14, p15,))
        pool.apply_async(conv2, (smilesY, p21, p22, p23, p24, p25,))
        pool.apply_async(conv3, (smilesB, p31, p32, p33, p34, p35,))
        pool.apply_async(conv4, (smilesB, p41, p42, p43, p44, p45,))
        pool.apply_async(conv5, (anjisuan, p51, p52, p53, p54, p55,))
        # 关闭进程池，不允许新的进程加入
        pool.close()
        # 运行进程池中的进程
        pool.join()
        # cout[0] = conv1(smilesY, p11, p12, p13, p14, p15)
        # cout[1] = conv2(smilesY, p21, p22, p23, p24, p25)
        # cout[2] = conv3(smilesB, p31, p32, p33, p34, p35)
        # cout[3] = conv4(smilesB, p41, p42, p43, p44, p45)
        # cout[4] = conv5(anjisuan, p51, p52, p53, p54, p55)
        # p1.start()
        # p2.start()
        # p3.start()
        # p4.start()
        # p5.start()
        # p1.join()
        # p2.join()0
        # p3.join()
        # p4.join()
        # p5.join()
        out = Qfu_conec_cirt(p61, p62, p63)
        out = round(out,2)
        y_pre = out * 24.650
        # print('预测的亲和力为：',y_pre)
        loss_value.append((affinity_truth[i] - y_pre) ** 2)
        end = time.time()
        print(str(round(end - start, 3)) + 's')
    print('总的损失为-------------------------------------4g(0.001)', np.mean(loss_value))
    return np.mean(loss_value)


para = np.array(b)
# 多进程必须在main函数内执行
if __name__ == '__main__':
    batch_size = 100
    random.shuffle(table.col_values(5, 1))
    random.shuffle(table.col_values(4, 1))
    random.shuffle(table.col_values(2, 1))
    smiles = table.col_values(5,1)
    anjisuan = table.col_values(4,1)
    affinity = table.col_values(2,1)
    #print(a)
    opt = GradientDescent(maxiter=2,learning_rate=0.001)
    #opt = COBYLA(rhobeg=1, maxiter=200)
    # opt_outcom1 = opt.minimize(loss_function,np.array(b))
    for e in range(100):
        smiles_valus = smiles[e*150:(e+1)*150]
        #smiles_valus = table.col_values(5, 1)
        anjisuan_values = anjisuan[e*150:(e+1)*150]
        #anjisuan_values = table.col_values(4, 1)
        affinity_truth = affinity[e*150:(e+1)*150]
        #affinity_truth = table.col_values(2, 1)
        #print(smiles_valus)
        #loss_function(b)
        print(opt.minimize(loss_function, para))

        #print(opt_outcom.x)
        #para = opt_outcom.x
        #print(para)
        #np.savetxt("para.txt", para, fmt='%f', delimiter=',')
        optpara1 = ','.join(str(i) for i in para)
        flie = open(r"para4g.txt", "a")
        flie.write(str(e) + '批次'+ '\n')
        flie.write(optpara1 + '\n')
        flie.close()
