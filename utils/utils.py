import os, time, datetime
import io, json, zlib

from collections import defaultdict

import numpy as np
import pandas as pd

from data_core import *
import jdfs


def read_raw_bytes(KEY, POOL, NS):
    
    bio = jdfs.get_raw_bytes(KEY, POOL, NS)
    bio.seek(0)
    bstr = bio.read()
    sobj = bstr.decode('UTF-8')
    file0 = io.StringIO(sobj)
    
    return file0

def read_raw_bytes_compressed(KEY, POOL, NS):

    bio = jdfs.get_raw_bytes(KEY, POOL, NS)
    bio = zlib.decompress(bio.read(), 16 + zlib.MAX_WBITS)
    bstr = io.BytesIO(bio).read()
    sobj = bstr.decode("UTF-8")
    file0 = io.StringIO(sobj)

    return file0


def process_order_log(tmp, average = True):
    
    tmp = tmp.sort_values(by = ['secid', 'internalId', 'updateType']).reset_index(drop = True)
    tmp = tmp.astype({'updateType': 'int',
                      'secid': 'int',
                      'internalId': 'int',
                      'finalState': 'int',
                      'orderDirection': 'int',
                      'inv_L': 'int',
                      'inv_S': 'int',
                      'orderPrice': 'float',
                      'tradePrice': 'float',
                      'absOrderSize': 'int64',
                      'absFilledThisUpdate': 'int64',
                      'absOrderSizeCumFilled': 'int64',
                      'clockAtArrival': 'int64'})
    for feat in ['aaa', 'mta', 'mrm', 'mrmum', 'mrrlma',
                 'mrsb90', 'mrss90',
                 'mrb100', 'mra100']:
        tmp[feat] = tmp[feat].astype(float)
    
    tmp['dateTime'] = tmp.caamd.astype(np.int64).apply(lambda x: datetime.fromtimestamp(x/1e6)).astype("datetime64[ns]")
    # tmp['dateTime'] = pd.to_datetime(tmp.caamd*1000)
    tmp['minute'] = tmp.dateTime.dt.hour*60\
                    + tmp.dateTime.dt.minute\
                    + (tmp.dateTime.dt.second + tmp.dateTime.dt.microsecond > 0).astype(int)
    
    tmp['minute'] = np.where(tmp.minute < 12.5*60,
                             tmp.minute - 9*60 - 30,
                             tmp.minute - 9*60 - 30 - 60)

    #
    tmp['orderNotional'] = tmp.orderPrice * tmp.absOrderSize
    tmp['tradeNotional'] = tmp.tradePrice * tmp.absFilledThisUpdate
    tmp['unfilledNotional'] = tmp.orderPrice * (tmp.absOrderSize - tmp.absOrderSizeCumFilled)

    #
    tmp['orderDirectionMod'] = np.where(tmp.updateType == 0,
                                        tmp.orderDirection,
                                        np.nan)
    tmp['orderDirectionMod'] = np.where((tmp.inv_L < tmp.inv_S) & (tmp.orderDirectionMod == 1),
                                        2,
                                        tmp.orderDirectionMod)
    tmp['orderDirectionMod'] = np.where((tmp.inv_L < tmp.inv_S) & (tmp.orderDirectionMod == -1),
                                        -2,
                                        tmp.orderDirectionMod)
    tmp['orderDirectionMod'] = tmp.groupby(['internalId', 'secid'])['orderDirectionMod'].fillna(method = 'ffill')

    #
    tmp['cashFlow'] = (-np.sign(tmp.orderDirectionMod) - 0.00155) * tmp.tradeNotional
    
    
    tmp['mid'] = (tmp.mra100 + tmp.mrb100) / 2 * .001
    tmp['spread'] = (tmp.mra100 - tmp.mrb100) * .001 / tmp.mid
    # tmp['theo'] = .5 * (tmp.mra100 * (1. + tmp.mrsb90) + tmp.mrb100 / (1. + tmp.mrss90)) * .001
    # tmp['smid90'] = tmp.theo / tmp.mid - 1.
    
    # tmp.rename({'mrsb300': 'netInvAdj',
    #             'mrss300': 'grossInvAdj'}, axis = 1, inplace = True)
    
    tn = tmp.groupby(['secid', 'internalId'])[['tradeNotional', 'cashFlow']].sum().reset_index()
    
    ## derivative
    insert = tmp[tmp.updateType == 0]

    if average:

        mediate = tmp[tmp.updateType == 4][['secid', 'internalId', 'absOrderSizeCumFilled', 'caamd', 'minute', 'unfilledNotional',
                                               'mid', "mrsb90", "mrss90", 'mrm', 'mrrlma', 'spread',
                                               'tradePrice', 'tradeNotional']]
        mediate['tradeNotional_SUM'] = mediate.groupby(['secid', 'internalId'])['tradeNotional'].transform('sum')
        
        for feat in ['absOrderSizeCumFilled', 'caamd', 'minute', 'unfilledNotional',
                                               'mid', "mrsb90", "mrss90", 'mrm', 'mrrlma', 'spread',
                                               'tradePrice']:
            mediate[feat] = mediate[feat] * mediate['tradeNotional'] / mediate['tradeNotional_SUM']

        final = mediate.groupby(['secid', 'internalId'])[['absOrderSizeCumFilled', 'caamd', 'minute', 'unfilledNotional',
                                               'mid', "mrsb90", "mrss90", 'mrm', 'mrrlma', 'spread',
                                               'tradePrice']].sum().reset_index()
        final['minute'] = final['minute'].astype(int)

    else:

         
        final = tmp[tmp.finalState == 1]\
                .groupby(['secid', 'internalId'])[['absOrderSizeCumFilled', 'caamd', 'minute', 'unfilledNotional',
                                                   'mrsb90', 'mrss90', 'mid', 'mrm', 'mrrlma', 'spread',
                                                   'tradePrice', 'mra100', 'mrb100']].last().reset_index()

    
    deriva = pd.merge(insert, final,
                      on = ['secid', 'internalId'],
                      how = 'left',
                      suffixes = ['', '_final'])
    deriva = pd.merge(deriva, tn,
                      on = ['secid', 'internalId'],
                      how = 'left',
                      suffixes = ['', '_final'])
    
    
    
    deriva['tradeNotional_final'] = deriva.tradeNotional_final.fillna(0)
    
    deriva['caamd_final'] = deriva.caamd_final.fillna(deriva.caamd)
    deriva['dateTime_final'] = deriva.caamd_final.astype(np.int64).apply(lambda x: datetime.fromtimestamp(x/1e6))
    # deriva['dateTime_final'] = pd.to_datetime(deriva.caamd_final*1000)
    deriva['sec_elapsed'] = np.clip((deriva.dateTime_final - deriva.dateTime).dt.total_seconds(), 0, np.inf)

    deriva['fill_rate'] = deriva.absOrderSizeCumFilled_final / deriva.absOrderSize
    
    
    deriva['weight_order'] = deriva.orderNotional / deriva.orderNotional.sum()
    deriva['weight_trade'] = deriva.tradeNotional_final / deriva.tradeNotional_final.sum()

    deriva["sta"] = np.where(
        deriva.orderDirection > 0,
        deriva.mrsb90,
        deriva.mrss90
    )
    deriva["sta_final"] = np.where(
        deriva.orderDirection > 0,
        deriva.mrsb90_final,
        deriva.mrss90_final
    )
    
    
    return tmp, deriva


def break_date(dd, year_break = False, concatenator = '-'):
    
    if year_break:
        strOut = str(dd)[:4]+'\n'+str(dd)[4:6]+concatenator+str(dd)[6:]
    else:
        strOut = str(dd)[:4]+concatenator+str(dd)[4:6]+concatenator+str(dd)[6:]
    
    return strOut

def process_barra_exposure(
    date,
    model,
    mta_nas = "",
):
    year = date // 10000
    
    if model[:3] == "CNE":
        _market = "chn"
        _d_headers = {
            "SH": 1000000,
            "SZ": 1000000,
        }
    elif model[:3] == "CXE":
        _market = "hkg"
        _d_headers = {
            "HK": 4000000,
        }
    elif model[:3] == "TWE":
        _market = "twn"
        _d_headers = {
            "TW": 10000000,
            "TW": 11000000,
        }
    elif model[:3] == "KRE":
        _market = "kor"
        _d_headers = {
            "KR": 12000000,
        }
    else:
        _market = ""
        _d_headers = {}
        
    # msci
    _dir = os.path.join(
        mta_nas,
        "rch/raw/msci/{}/daily/csv/{}/".format(
            model,
            year
        )
    )
    _fname = os.path.join(
        _dir,
        "{}_100_Asset_Exposure.{}.csv.gz".format(
            model,
            break_date(date)
        )
    )
    _exp = pd.read_csv(_fname)
    
    #
    _exp_0 = _exp[pd.notnull(_exp.skey)]
    _exp_full = []
    for _key, _mkt_id in _d_headers.items():
        _exp = _exp_0[
            (_exp_0.skey.str[:2] == _key) &
            (_exp_0.skey.str[2:].str.isnumeric())
        ].copy()
        
        _exp["secid"] = _exp.skey.str[2:].astype(int) + _mkt_id
        _exp_full.append(_exp)
    _exp_full = pd.concat(_exp_full)
    
    _exp_full = _exp_full.groupby([
        "secid", "Factor"
    ])[["Exposure"]].first().reset_index()
    
    return _exp_full

def _ifr_calculator(_df):
    _denom = np.maximum(
        _df.groupby(["secid"])[["inv_net_aft_lsz"]].max(),
        _df.groupby(["secid"])[["inv_net"]].max()
    ) - np.minimum(
        _df.groupby(["secid"])[["inv_net_aft_lsz"]].min(),
        _df.groupby(["secid"])[["inv_net"]].min()
    )
    _numer = _df.groupby(["secid"])[["inv_net"]].max() - \
        _df.groupby(["secid"])[["inv_net"]].min()
    _wgt = _df.groupby(["secid"])[["price"]].mean()
    _calc = _denom.join(_numer)
    _calc = _calc.join(_wgt)
    _ifr = (_calc.inv_net * _calc.price).sum() / \
        (_calc.inv_net_aft_lsz * _calc.price).sum()
    
    return _ifr

def remove_log_timestamp(_path, _dtype, _pos_date):
    
    if _path[-1] != "/":
        _path = _path + "/"
    
    try:
        _list_logs = jdfs.list_dir(
            _path,
            _path.split("/")[1],
            _path.split("/")[2]
        )
        _suffix = "csv.gz"

        for _kname in [_f for _f in _list_logs if _dtype in _f]:
            assert(_suffix in _kname)
            _date = int(_kname.split("/")[-1].split("_")[_pos_date])

            _df = jdfs.read_file(
                _kname,
                _kname.split("/")[1],
                _kname.split("/")[2]
            )

            _strout = \
                "{}_{}.{}"\
                .format(_dtype, _date, _suffix)
            _oname = \
                "{}{}"\
                .format(_path, _strout)
            # print(_oname)

            jdfs.write_df_to_jdfs(
                _df,
                _oname,
                _oname.split("/")[1],
                _oname.split("/")[2]
            )
            
        return True
    
    except Exception as _e:
        print(_e)
        return False


def find_sec_name(date, ddict):
    _found = False
    for _ikey, _ival in ddict.items():
        if date in _ival[0]:
            _found = True
            break
            
    if _found:
        return _ikey
    else:
        return ""

def summarize_po(
    _d_po,
    _ndays_rct = 120,
):
    _summary = pd.DataFrame()

    _tmp = _d_po.groupby(["run_num"])[[
        "fill_rate_intraday", "fill_rate_long", "fill_rate_short"
    ]].mean().reset_index()

    _summary["run_num"] = _tmp.run_num
    _summary["fill_rate_intraday_pct"] = _tmp.fill_rate_intraday * 100
    _summary["fill_rate_long_pct"] = _tmp.fill_rate_long * 100
    _summary["fill_rate_short_pct"] = _tmp.fill_rate_short * 100

    _tmp = _d_po.groupby(["run_num"]).tail(_ndays_rct).groupby(["run_num"])[[
        "fill_rate_intraday", "fill_rate_long", "fill_rate_short"
    ]].mean().reset_index()

    _summary["fill_rate_intraday_pct_rct{}".format(_ndays_rct)] = \
        _tmp.fill_rate_intraday * 100
    _summary["fill_rate_long_pct_rct{}".format(_ndays_rct)] = \
        _tmp.fill_rate_long * 100
    _summary["fill_rate_short_pct_rct{}".format(_ndays_rct)] = \
        _tmp.fill_rate_short * 100

    return _summary


def summarize(
    _d_df,
    comm,
    _ndays = 243,
    _ndays_rct = 120,
    _turnover_columns = ["turnover_long", "turnover_sell", "turnover_short"],
    _ret_idx = None
):
    _summary = pd.DataFrame()

    _d_df["win"] = _d_df.ret_raw > 0
    _tmp = _d_df.groupby(["run_num"])["ret_raw"].describe().reset_index()
    _tmp_sep = _d_df.groupby(["run_num"])[["ret_holding", "ret_trading"]].describe().reset_index()
    _tmp_orders = \
        _d_df.groupby(["run_num"])[["turnover"] + _turnover_columns].describe().reset_index()
    _summary["run_num"] = _tmp.run_num
    _summary["net_ret_bps"] = _tmp["mean"] * 10000
    _summary["net_ret_holding_bps"] = \
        _tmp_sep.loc[:, ("ret_holding", "mean")] * 10000
    _summary["net_ret_trading_bps"] = \
        _tmp_sep.loc[:, ("ret_trading", "mean")] * 10000
    _eod_lmv_ratio_pct = \
        _d_df.groupby(["run_num"])["eod_lmv_ratio"].mean().values * 100
    _eod_smv_ratio_pct = \
        _d_df.groupby(["run_num"])["eod_smv_ratio"].mean().values * 100
    _summary["mv_ratio_pct"] = \
        .5 * (_eod_lmv_ratio_pct + _eod_smv_ratio_pct) 
    _summary["turnover_pct"] = \
        _tmp_orders.loc[:, ("turnover", "mean")] * 100
    _summary["ret_per_rt_bps"] = \
        (_summary.net_ret_bps * .0001 / (_summary.turnover_pct * .01) + comm) * 10000
    _summary["mv_ratio_long_pct"] = _eod_lmv_ratio_pct
    _summary["mv_ratio_short_pct"] = _eod_smv_ratio_pct
    for _col in _turnover_columns:
        _summary["{}_pct".format(_col)] = \
            _tmp_orders.loc[:, (_col, "mean")] * 100
    _summary["win_ratio_pct"] = \
        _d_df.groupby(["run_num"]).win.sum().values / \
        _d_df.groupby(["run_num"]).size().values * 100

    _summary["risk_pct"] = _tmp["std"] * np.sqrt(_ndays) * 100
    _summary["risk_holding_pct"] = \
        _tmp_sep.loc[:, ("ret_holding", "std")] * np.sqrt(_ndays) * 100
    _summary["risk_trading_pct"] = \
        _tmp_sep.loc[:, ("ret_trading", "std")] * np.sqrt(_ndays) * 100
    _summary["sharpe"] = _tmp["mean"] / _tmp["std"] * np.sqrt(_ndays)

    # MDD
    def cal_mdd(_x):
        cum_ret = np.cumsum(_x)
        dd = np.maximum.accumulate(1 + cum_ret) - (1 + cum_ret)
        mdd = np.max(dd)

        return mdd
    _summary["mdd_pct"] = (_d_df.groupby(["run_num"])["ret_raw"].apply(cal_mdd) * 100).values
    _summary["min_daily_bps"] = _tmp["min"] * 10000

    # beta
    if _ret_idx is not None:
        _df = pd.merge(
            _d_df[["date", "run_num", "ret_raw", "ret_holding", "ret_trading"]],
            _ret_idx[["date", "daily_return"]],
            on = ["date"],
            how = "left"
        )

        _cov = _df.groupby(["run_num"])[["daily_return", "ret_raw", "ret_holding", "ret_trading"]].cov()

        _dict_beta = defaultdict(list)
        for _run_num in _cov.index.unique(level = "run_num"):
            for _ret in ["ret_raw", "ret_holding", "ret_trading"]:
                _beta = _cov.loc[(_run_num, "daily_return"), _ret] / _cov.loc[(_run_num, "daily_return"), "daily_return"]
                _dict_beta[_ret].append(_beta)

        _summary["beta"] = _dict_beta["ret_raw"]
        _summary["beta_holding"] = _dict_beta["ret_holding"]
        _summary["beta_trading"] = _dict_beta["ret_trading"]

    else:
        _summary["beta"] = np.nan
        _summary["beta_holding"] = np.nan
        _summary["beta_trading"] = np.nan
    
    # recent 63 days
    _tmp_rct = \
        _d_df.groupby(["run_num"]).tail(_ndays_rct).groupby(["run_num"])["ret_raw"].describe().reset_index()
    _tmp_orders = \
        _d_df.groupby(["run_num"]).tail(_ndays_rct)\
        .groupby(["run_num"])[["turnover"] + _turnover_columns].describe().reset_index()
    _summary["net_ret_bps_rct{}".format(_ndays_rct)] = _tmp_rct["mean"] * 10000
    _summary["turnover_pct_rct{}".format(_ndays_rct)] = \
        _tmp_orders.loc[:, ("turnover", "mean")] * 100
    for _col in _turnover_columns:
        _summary["{}_pct_rct{}".format(_col, _ndays_rct)] = \
            _tmp_orders.loc[:, (_col, "mean")] * 100
    _eod_lmv_ratio_pct_rct = \
        _d_df.groupby(["run_num"]).tail(_ndays_rct).groupby(["run_num"])["eod_lmv_ratio"].mean().values * 100
    _eod_smv_ratio_pct_rct = \
        _d_df.groupby(["run_num"]).tail(_ndays_rct).groupby(["run_num"])["eod_smv_ratio"].mean().values * 100
    _summary["mv_ratio_pct_rct{}".format(_ndays_rct)] = \
        .5 * (_eod_lmv_ratio_pct_rct + _eod_smv_ratio_pct_rct)
    _summary["ret_per_rt_bps_rct{}".format(_ndays_rct)] = \
        (_summary["net_ret_bps_rct{}".format(_ndays_rct)] * .0001 / \
        (_summary["turnover_pct_rct{}".format(_ndays_rct)] * .01) + comm) * 10000
    _summary["win_ratio_pct_rct{}".format(_ndays_rct)] = \
        _d_df.groupby(["run_num"]).tail(_ndays_rct).groupby(["run_num"]).win.sum().values / \
        _d_df.groupby(["run_num"]).tail(_ndays_rct).groupby(["run_num"]).size().values * 100
    _summary["risk_pct_rct{}".format(_ndays_rct)] = _tmp_rct["std"] * np.sqrt(_ndays) * 100
    _summary["sharpe_rct{}".format(_ndays_rct)] = _tmp_rct["mean"] / _tmp_rct["std"] * np.sqrt(_ndays)
    
    return _summary

def derive_minutes(df, market):
    if market == "hkg":
        df["mins_since_open"] = np.where(
            df.minuteCreated >= 1215,
            (
                df.minuteCreated // 100 * 60 + df.minuteCreated % 100 - \
                (9*60 + 30 - 60)
            ),
            (
                df.minuteCreated // 100 * 60 + df.minuteCreated % 100 - \
                (9*60 + 30)
            )
        )
    elif market in ["twn", "kor"]:
        df["mins_since_open"] = (
            df.minuteCreated // 100 * 60 + df.minuteCreated % 100 - \
            9*60
        )
        
    return df
