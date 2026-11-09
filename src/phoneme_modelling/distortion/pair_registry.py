"""Every sound-pair distorter, plain and matched, so the pipeline can register them in one loop.

One module per pair lives in ``distortion/distorters/``; add both of a new pair's classes here.
"""

from phoneme_modelling.distortion.distorters.b_to_p import BToPDistorter, BToPMatchedDistorter
from phoneme_modelling.distortion.distorters.ch_to_k import ChToKDistorter, ChToKMatchedDistorter
from phoneme_modelling.distortion.distorters.ch_to_sh import ChToShDistorter, ChToShMatchedDistorter
from phoneme_modelling.distortion.distorters.ch_to_t import ChToTDistorter, ChToTMatchedDistorter
from phoneme_modelling.distortion.distorters.d_to_g import DToGDistorter, DToGMatchedDistorter
from phoneme_modelling.distortion.distorters.d_to_k import DToKDistorter, DToKMatchedDistorter
from phoneme_modelling.distortion.distorters.d_to_t import DToTDistorter, DToTMatchedDistorter
from phoneme_modelling.distortion.distorters.f_to_b import FToBDistorter, FToBMatchedDistorter
from phoneme_modelling.distortion.distorters.f_to_p import FToPDistorter, FToPMatchedDistorter
from phoneme_modelling.distortion.distorters.f_to_v import FToVDistorter, FToVMatchedDistorter
from phoneme_modelling.distortion.distorters.g_to_d import GToDDistorter, GToDMatchedDistorter
from phoneme_modelling.distortion.distorters.g_to_k import GToKDistorter, GToKMatchedDistorter
from phoneme_modelling.distortion.distorters.j_to_d import JToDDistorter, JToDMatchedDistorter
from phoneme_modelling.distortion.distorters.j_to_sh import JToShDistorter, JToShMatchedDistorter
from phoneme_modelling.distortion.distorters.j_to_z import JToZDistorter, JToZMatchedDistorter
from phoneme_modelling.distortion.distorters.k_to_g import KToGDistorter, KToGMatchedDistorter
from phoneme_modelling.distortion.distorters.k_to_t import KToTDistorter, KToTMatchedDistorter
from phoneme_modelling.distortion.distorters.l_to_w import LToWDistorter, LToWMatchedDistorter
from phoneme_modelling.distortion.distorters.l_to_y import LToYDistorter, LToYMatchedDistorter
from phoneme_modelling.distortion.distorters.m_to_b import MToBDistorter, MToBMatchedDistorter
from phoneme_modelling.distortion.distorters.m_to_n import MToNDistorter, MToNMatchedDistorter
from phoneme_modelling.distortion.distorters.n_to_d import NToDDistorter, NToDMatchedDistorter
from phoneme_modelling.distortion.distorters.n_to_m import NToMDistorter, NToMMatchedDistorter
from phoneme_modelling.distortion.distorters.n_to_t import NToTDistorter, NToTMatchedDistorter
from phoneme_modelling.distortion.distorters.ng_to_g import NgToGDistorter, NgToGMatchedDistorter
from phoneme_modelling.distortion.distorters.ng_to_n import NgToNDistorter, NgToNMatchedDistorter
from phoneme_modelling.distortion.distorters.p_to_b import PToBDistorter, PToBMatchedDistorter
from phoneme_modelling.distortion.distorters.r_distortion import RDistortionDistorter, RDistortionMatchedDistorter
from phoneme_modelling.distortion.distorters.r_to_l import RToLDistorter, RToLMatchedDistorter
from phoneme_modelling.distortion.distorters.r_to_w import RToWDistorter, RToWMatchedDistorter
from phoneme_modelling.distortion.distorters.r_to_y import RToYDistorter, RToYMatchedDistorter
from phoneme_modelling.distortion.distorters.s_to_d import SToDDistorter, SToDMatchedDistorter
from phoneme_modelling.distortion.distorters.s_to_t import SToTDistorter, SToTMatchedDistorter
from phoneme_modelling.distortion.distorters.s_to_th_voiced import SToThVoicedDistorter, SToThVoicedMatchedDistorter
from phoneme_modelling.distortion.distorters.s_to_th_voiceless import (
    SToThVoicelessDistorter,
    SToThVoicelessMatchedDistorter,
)
from phoneme_modelling.distortion.distorters.s_to_z import SToZDistorter, SToZMatchedDistorter
from phoneme_modelling.distortion.distorters.sh_to_ch import ShToChDistorter, ShToChMatchedDistorter
from phoneme_modelling.distortion.distorters.sh_to_s import ShToSDistorter, ShToSMatchedDistorter
from phoneme_modelling.distortion.distorters.sh_to_t import ShToTDistorter, ShToTMatchedDistorter
from phoneme_modelling.distortion.distorters.t_to_d import TToDDistorter, TToDMatchedDistorter
from phoneme_modelling.distortion.distorters.t_to_g import TToGDistorter, TToGMatchedDistorter
from phoneme_modelling.distortion.distorters.t_to_k import TToKDistorter, TToKMatchedDistorter
from phoneme_modelling.distortion.distorters.th_voiced_to_d import ThVoicedToDDistorter, ThVoicedToDMatchedDistorter
from phoneme_modelling.distortion.distorters.th_voiced_to_th_voiceless import (
    ThVoicedToThVoicelessDistorter,
    ThVoicedToThVoicelessMatchedDistorter,
)
from phoneme_modelling.distortion.distorters.th_voiced_to_v import ThVoicedToVDistorter, ThVoicedToVMatchedDistorter
from phoneme_modelling.distortion.distorters.th_voiceless_to_f import (
    ThVoicelessToFDistorter,
    ThVoicelessToFMatchedDistorter,
)
from phoneme_modelling.distortion.distorters.th_voiceless_to_s import (
    ThVoicelessToSDistorter,
    ThVoicelessToSMatchedDistorter,
)
from phoneme_modelling.distortion.distorters.th_voiceless_to_t import (
    ThVoicelessToTDistorter,
    ThVoicelessToTMatchedDistorter,
)
from phoneme_modelling.distortion.distorters.v_to_b import VToBDistorter, VToBMatchedDistorter
from phoneme_modelling.distortion.distorters.v_to_f import VToFDistorter, VToFMatchedDistorter
from phoneme_modelling.distortion.distorters.w_to_r import WToRDistorter, WToRMatchedDistorter
from phoneme_modelling.distortion.distorters.y_to_w import YToWDistorter, YToWMatchedDistorter
from phoneme_modelling.distortion.distorters.z_to_d import ZToDDistorter, ZToDMatchedDistorter
from phoneme_modelling.distortion.distorters.z_to_s import ZToSDistorter, ZToSMatchedDistorter
from phoneme_modelling.distortion.distorters.z_to_t import ZToTDistorter, ZToTMatchedDistorter
from phoneme_modelling.distortion.distorters.z_to_th_voiced import ZToThVoicedDistorter, ZToThVoicedMatchedDistorter
from phoneme_modelling.distortion.distorters.z_to_th_voiceless import (
    ZToThVoicelessDistorter,
    ZToThVoicelessMatchedDistorter,
)
from phoneme_modelling.distortion.pair_distorter import PairDistorter

PAIR_DISTORTERS: list[type[PairDistorter]] = [
    BToPDistorter,
    BToPMatchedDistorter,
    ChToKDistorter,
    ChToKMatchedDistorter,
    ChToShDistorter,
    ChToShMatchedDistorter,
    ChToTDistorter,
    ChToTMatchedDistorter,
    DToGDistorter,
    DToGMatchedDistorter,
    DToKDistorter,
    DToKMatchedDistorter,
    DToTDistorter,
    DToTMatchedDistorter,
    FToBDistorter,
    FToBMatchedDistorter,
    FToPDistorter,
    FToPMatchedDistorter,
    FToVDistorter,
    FToVMatchedDistorter,
    GToDDistorter,
    GToDMatchedDistorter,
    GToKDistorter,
    GToKMatchedDistorter,
    JToDDistorter,
    JToDMatchedDistorter,
    JToShDistorter,
    JToShMatchedDistorter,
    JToZDistorter,
    JToZMatchedDistorter,
    KToGDistorter,
    KToGMatchedDistorter,
    KToTDistorter,
    KToTMatchedDistorter,
    LToWDistorter,
    LToWMatchedDistorter,
    LToYDistorter,
    LToYMatchedDistorter,
    MToBDistorter,
    MToBMatchedDistorter,
    MToNDistorter,
    MToNMatchedDistorter,
    NToDDistorter,
    NToDMatchedDistorter,
    NToMDistorter,
    NToMMatchedDistorter,
    NToTDistorter,
    NToTMatchedDistorter,
    NgToGDistorter,
    NgToGMatchedDistorter,
    NgToNDistorter,
    NgToNMatchedDistorter,
    PToBDistorter,
    PToBMatchedDistorter,
    RDistortionDistorter,
    RDistortionMatchedDistorter,
    RToLDistorter,
    RToLMatchedDistorter,
    RToWDistorter,
    RToWMatchedDistorter,
    RToYDistorter,
    RToYMatchedDistorter,
    SToDDistorter,
    SToDMatchedDistorter,
    SToTDistorter,
    SToTMatchedDistorter,
    SToThVoicedDistorter,
    SToThVoicedMatchedDistorter,
    SToThVoicelessDistorter,
    SToThVoicelessMatchedDistorter,
    SToZDistorter,
    SToZMatchedDistorter,
    ShToChDistorter,
    ShToChMatchedDistorter,
    ShToSDistorter,
    ShToSMatchedDistorter,
    ShToTDistorter,
    ShToTMatchedDistorter,
    TToDDistorter,
    TToDMatchedDistorter,
    TToGDistorter,
    TToGMatchedDistorter,
    TToKDistorter,
    TToKMatchedDistorter,
    ThVoicedToDDistorter,
    ThVoicedToDMatchedDistorter,
    ThVoicedToThVoicelessDistorter,
    ThVoicedToThVoicelessMatchedDistorter,
    ThVoicedToVDistorter,
    ThVoicedToVMatchedDistorter,
    ThVoicelessToFDistorter,
    ThVoicelessToFMatchedDistorter,
    ThVoicelessToSDistorter,
    ThVoicelessToSMatchedDistorter,
    ThVoicelessToTDistorter,
    ThVoicelessToTMatchedDistorter,
    VToBDistorter,
    VToBMatchedDistorter,
    VToFDistorter,
    VToFMatchedDistorter,
    WToRDistorter,
    WToRMatchedDistorter,
    YToWDistorter,
    YToWMatchedDistorter,
    ZToDDistorter,
    ZToDMatchedDistorter,
    ZToSDistorter,
    ZToSMatchedDistorter,
    ZToTDistorter,
    ZToTMatchedDistorter,
    ZToThVoicedDistorter,
    ZToThVoicedMatchedDistorter,
    ZToThVoicelessDistorter,
    ZToThVoicelessMatchedDistorter,
]
