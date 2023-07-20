import os, sys, json, time
import multiprocessing

import numpy as np
import pandas as pd

from data_core import constructor_set_region_asset, GenericTabularData
import jdfs

import white_simv2_light.utils.utils as utils
from white_simv2_light.utils.helper import *
import white_simv2_light.utils.logger as logger
logging = logger.Logger("Lighter", level = "info")

import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)


class Lighter(object):
    
    '''
    Class object for loading and running postsim.
    
    Inputs
    ------
    market : str
    The market/region the sim is for.

    run_prefix : str
    The dfs key prefix, including a variable that contains the sec
    name that corresponds to the keys in the date dictionary.

    date_dict : str
    The date dictionary that hosts: {key: [[dates], [securities]]}.
    The key corresponds to the section name used in run_prefix.
    TODO Add a util func for parsing this.

    pool : str, optional
    The dfs pool.  Assumed to be the first subdir in run_prefix,
    if not provided.

    ns : str, optional
    The dfs namespace.  Assumed to be the second subdir in run_prefix,
    if not provided.

    mp_pool_size : int, optional
    The number of processes when running multiprocessing.
    Default is 8.

    compress : bool, optional
    The "compress" setting in the config "logToCeph" part.
    Default is True.

    mta_nas : str, optional
    The directory to the mta_nas folder.  Used only when deriving the exposure.
    Default is "".

    cap : float, optional
    The single-side target capital.  Used only when deriving the exposure.
    Default is -1.
    
    '''
    
    def __init__(
        self,
        market: str,
        run_prefix: str,
        date_dict: str,
        pool: str = "",
        ns: str = "",
        mp_pool_size : int = 8,
        compress : bool = True,
        mta_nas : str = "",
        cap : float = -1.,
    ):
        self._mkt = market.lower()
        self._rprefix = run_prefix
        with open(date_dict) as _f:
            self._ddict = json.load(_f)
        self._ddict_full = {} # filled in prepare()
        if not (pool and ns):
            self._pool = self._rprefix.split("/")[1]
            self._ns = self._rprefix.split("/")[2]
        else:
            self._pool = pool
            self._ns = ns

        if self._mkt == "hkg":
            self._univ = "HKGUniv.EqTOP500"
            self._bm = "HSI"
            self._barra_model = "CXE1S"
        elif self._mkt == "twn":
            self._univ = "TWNUniv.EqTOP600"
            self._bm = "TAIEX"
            self._barra_model = "TWE2D"
        elif self._mkt == "kor":
            self._univ = "KORUniv.EqTOP800"
            self._bm = "KS200"
            self._barra_model = "KRE3D"
        else:
            self._univ = ""
            self._bm = ""
            self._barra_model = ""

        self._ra = constructor_set_region_asset(market, "eq")
        self._date_list = []
        for _ikey, _ival in self._ddict.items():
            self._date_list = list(set(self._date_list) | set(_ival[0]))
        self._date_list = sorted(set(self._date_list))
            
        logging.info("[init] Loaded mkt %s", self._mkt)
        logging.info("[init] rprefix %s pool %s ns %s", self._rprefix, self._pool, self._ns)
        logging.info("[init] date_dict %s", date_dict)
        logging.info(
            "[init] A total of %i days from %s to %s",
            len(self._date_list),
            utils.break_date(min(self._date_list)),
            utils.break_date(max(self._date_list))
        )
       
        # Multiprocessing pool size 
        self._mpn = mp_pool_size

        # Compressed
        self._compressed = compress

        # mta nas
        self._mta_nas = mta_nas

        # Single-side capital used to derive exposure weight
        self._cap = cap
        
        # Prevent reloading
        self._s_bar1m = None
        self._s_eod = None
        self._s_beta = None

        self._m_orders = None
        self._m_orders_by_dir = None

        self._d_inventory = None

        self._m_po = None
        self._d_po = None
        self._m_po_inv = None

        self._d_exp = None

    def prepare(
        self,
        force = False
    ):
        '''
        Get the data_core data.
        '''

        if (not force) and self._s_bar1m and self._s_eod and self._s_beta:
            logging.info(
                "[prepare] Skip; tables already exist.  Use force = True to force reload."
            )

        else:
            logging.warning(
                "[prepare] Estimated to take %.1f minutes",
                0.3*len(self._ddict.keys())
            )

            _s_bar1m = {}
            _s_eod = {}
            _s_beta = {}
            _last_key = sorted([_k for _k in self._ddict.keys()])[-1]
            for _ikey, _ival in self._ddict.items():

                time0 = time.time()

                _max_date = \
                    self._ra.dates.shift_trading_dates(max(_ival[0]), 1)

                _cfg = GenericTabularData.ConfigParams()
                _cfg.vars = \
                    ["yyyymmdd", "jkey", "hhmmss_nano", "prvi_last", "amount", "volume"]
                _s_bar1m[_ikey] = GenericTabularData(
                    region = self._mkt,
                    asset = "eq",
                    dataset = "md_bar1m",
                    univ = _ival[1],
                    start_date = min(_ival[0]),
                    end_date = _max_date,
                    config_params = _cfg
                ).as_data_frame()
                _s_bar1m[_ikey]["date"] = _s_bar1m[_ikey]["yyyymmdd"]
                _s_bar1m[_ikey]["secid"] = _s_bar1m[_ikey]["jkey"]
                _s_bar1m[_ikey]["minute"] = _s_bar1m[_ikey].hhmmss_nano // 100000000000
                _s_bar1m[_ikey] = _s_bar1m[_ikey].sort_values(by = ["date", "secid", "minute"])
                _s_bar1m[_ikey]["vwap"] = _s_bar1m[_ikey].amount / _s_bar1m[_ikey].volume
                _s_bar1m[_ikey]["vwap"] = \
                    _s_bar1m[_ikey].groupby(["date", "secid"])["vwap"].fillna(method = "ffill")

                _cfg = GenericTabularData.ConfigParams()
                _cfg.vars = \
                    ["jkey", "yyyymmdd", "round_lot_size",
                     "suspended", "open", "close", "prvd_close_adj"] + \
                    (["adj_factor"] if (self._mkt == "kor") else [])
                _s_eod[_ikey] = GenericTabularData(
                    region = self._mkt,
                    asset = "eq",
                    dataset = "md_eod",
                    univ = _ival[1],
                    start_date = min(_ival[0]),
                    end_date = _max_date,
                    config_params = _cfg
                ).as_data_frame()
                _s_eod[_ikey]["date"] = _s_eod[_ikey]["yyyymmdd"]
                _s_eod[_ikey]["secid"] = _s_eod[_ikey]["jkey"]
                if self._mkt == "kor":
                    logging.warning("[prepare] Applied adj_factor on prvd_close_adj for %s", self._mkt)
                    _s_eod[_ikey]["prvd_close_adj"] = \
                        _s_eod[_ikey].prvd_close_adj / _s_eod[_ikey].adj_factor

                _cfg = GenericTabularData.ConfigParams()
                _cfg.vars = \
                    ["jkey", "yyyymmdd", "beta_{}".format(self._bm)]
                _s_beta[_ikey] = GenericTabularData(
                    region = self._mkt,
                    asset = "eq",
                    dataset = "ms_mkt_beta",
                    univ = _ival[1],
                    start_date = min(_ival[0]),
                    end_date = _max_date,
                    config_params = _cfg
                ).as_data_frame()
                _s_beta[_ikey]["date"] = _s_beta[_ikey]["yyyymmdd"]
                _s_beta[_ikey]["secid"] = _s_beta[_ikey]["jkey"]

                logging.info("[prepare] %s done; it took %.2f seconds.", _ikey, time.time() - time0)

            self._s_bar1m = _s_bar1m
            self._s_eod = _s_eod
            self._s_beta = _s_beta

            for _ikey, _idf in self._s_eod.items():
                _entry = [
                    sorted(_idf.date.unique()),
                    sorted(_idf.secid.unique())
                ]
                self._ddict_full[_ikey] = _entry

    def process_orders(
        self,
        force = False
    ):
        '''
        Process orders/trades.

        Inputs
        ------
        force : bool
        Force reload.
        '''
        # No direction grouping
        if force or (self._m_orders is None):
            logging.warning(
                "[process_orders] All dirs; estimated to take %.1f minutes.",
                1.85 * len(self._date_list) / 60
            )

            time0 = time.time()
            
            @pool_this(n_processes = self._mpn)
            def _process_orders_all(_date):
                try:
                    result = self._get_order_details(_date)
                    return result

                except Exception as _e:
                    print(_e)
                    return pd.DataFrame()


            self._m_orders = pd.concat(_process_orders_all(self._date_list))
            self._m_orders = self._m_orders.sort_values(
                by = ["date", "minute"]
            ).reset_index(drop = True)

            logging.info(
                "[process_orders] %i of %i days loaded; it took %.2f seconds.",
                len(self._m_orders.date.unique()),
                len(self._date_list),
                time.time() - time0
            )

        else:
            logging.info("[process_orders] Skip; table(s) already exist.  Use force = True to force reload.")

    def process_orders_by_dir(
        self,
        force = False
    ):
        '''
        Process orders/trades by orderDirection.

        Inputs
        ------
        force : bool
        Force reload.
        '''

        ## SKIP ALTOGETHER FOR NOW
        if force or (self._m_orders_by_dir is None):
            logging.warning(
                "[process_orders] By dir; estimated to take %.1f minutes.",
                1.9 * len(self._date_list) / 60
            )

            time0 = time.time()

            @pool_this(n_processes = self._mpn)
            def _process_orders_by_dir(_date):
                try:
                    result = self._get_order_details(_date, extra_grouping = ["orderDirection"])
                    return result

                except Exception as _e:
                    print(_e)
                    return pd.DataFrame()


            self._m_orders_by_dir = pd.concat(_process_orders_by_dir(self._date_list))
            self._m_orders_by_dir = self._m_orders_by_dir.sort_values(
                by = ["date", "minute", "orderDirection"]
            ).reset_index(drop = True)

            logging.info(
                "[process_orders] %i of %i days loaded; it took %.2f seconds.",
                len(self._m_orders_by_dir.date.unique()),
                len(self._date_list),
                time.time() - time0
            )

        else:
            _str_to_log = \
                "; table exists." if (self._m_orders_by_dir is not None) else ""
            logging.info(
                "[process_orders] Skip%s  Use force = True to force reload.",
                _str_to_log
            )

    def process_orders_fast(
        self,
        force = False
    ):
        '''
        Fast process orders/trades, using the fast method; NOT including participation and
        improvements over vwap.

        Inputs
        ------
        force : bool
        Force reload.
        '''
        
        if force or (self._m_orders is None):
            logging.warning(
                "[process_orders_fast] Fast processing of order details; estimated to take %.1f minutes." + \
                "\n[process_orders_fast] NOT including participation and improvements over vwap.",
                0.91 * len(self._date_list) / 60
            )

            time0 = time.time()

            @pool_this(n_processes = self._mpn)
            def _process_orders_fast(_date):
                try:
                    result = self._get_order_details_fast(_date, extra_grouping = ["orderDirection"])
                    return result

                except Exception as _e:
                    print(_e)
                    return pd.DataFrame()


            self._m_orders = pd.concat(_process_orders_fast(self._date_list))
            self._m_orders = self._m_orders.sort_values(
                by = ["date", "minute", "orderDirection"]
            ).reset_index(drop = True)

            if self._m_orders_by_dir is None:
                self._m_orders_by_dir = self._m_orders.copy()

            logging.info(
                "[process_orders_fast] %i of %i days loaded; it took %.2f seconds.",
                len(self._m_orders.date.unique()),
                len(self._date_list),
                time.time() - time0
            )

        else:
            logging.warning(
                "[process_orders_fast] Skip because the table exists; use force = True to force a reload."
            )


    def process_inventory(
        self,
        force = False
    ):
        '''
        Process inventory/cash.

        Inputs
        ------
        force : bool
        Force reload.
        '''
        
        if force or (self._d_inventory is None):
            logging.warning(
                "[process_inventory] Processing of inventory/cash details; estimated to take %.1f minutes.",
                0.027 * len(self._date_list) / 60
            )

            time0 = time.time()

            @pool_this(n_processes = self._mpn)
            def _process_inventory(_date):
                try:
                    result = self._get_inventory_details(_date)
                    return result

                except Exception as _e:
                    print(_e)
                    return pd.DataFrame()


            self._d_inventory = pd.concat(_process_inventory(self._date_list))
            self._d_inventory = self._d_inventory.sort_values(
                by = ["date"]
            ).reset_index(drop = True)

            logging.info(
                "[process_inventory] %i of %i days loaded; it took %.2f seconds.",
                len(self._d_inventory.date.unique()),
                len(self._date_list),
                time.time() - time0
            )

        else:
            logging.warning(
                "[process_inventory] Skip because the table exists; use force = True to force a reload."
            )

    def process_inventory_test(
        self
    ):
        time0 = time.time()

        def _process_inventory(_date, *args):
            _df = self._get_inventory_details(_date)
            return _df
        _func_to_wrap = my_decorator(_process_inventory)
        
        pool = multiprocessing.Pool(8)
        wrapped = [pool.apply_async(_func_to_wrap, (_d, {})) for _d in self._date_list]
        pool.close()

        self._d_inventory = pd.concat([_r.get() for _r in wrapped])
        self._d_inventory = self._d_inventory.sort_values(
            by = ["date"]
        ).reset_index(drop = True)

        logging.info(
            "[process_inventory] %i of %i days loaded; it took %.2f seconds.",
            len(self._d_inventory.date.unique()),
            len(self._date_list),
            time.time() - time0
        )


    # support funcs
    def _get_order_details(
        self,
        date,
        extra_grouping = []
    ):
        '''
        A support function that derives the minutely order details,
        including participation and improvement over vwap.

        Inputs
        ------
        date : int
        Date.

        extra_grouping : list of str, optional
        Needs to be column names of the order log.  Most commonly, "orderDirection".
        Default is an empty list [].
        '''
        _sec_name = utils.find_sec_name(date, self._ddict)
        if not _sec_name:
            return pd.DataFrame()

        try:
            kprefix = \
                self._rprefix\
                .format(_sec_name)

            _d = jdfs.read_file(
                os.path.join(kprefix, "orderLog_{}.csv{}".format(date, ".gz" if self._compressed else "")),
                self._pool,
                self._ns
            )
            _d0, _d = utils.process_order_log(_d)
            #
            _d["minute"] = _d.dateTime.dt.hour * 100 + _d.dateTime.dt.minute
            _d_notional = \
                _d.groupby(list(set(["minute", "secid"] + extra_grouping)))[[
                    "orderNotional", "tradeNotional_final", "absOrderSize"
                ]].sum().reset_index()
            _d_alpha = _d.groupby(list(set(["minute", "secid"] + extra_grouping)))[[
                "sta", "sta_final", "mrm", "mrm_final"
            ]].mean().reset_index()
            _d = pd.merge(
                _d_notional,
                _d_alpha,
                on = list(set(["minute", "secid"] + extra_grouping)),
                how = "left"
            )
            _d["date"] = date
            _d["orderVwap"] = _d.orderNotional / _d.absOrderSize

            #
            _d0 = _d0[_d0.updateType == 4]
            _d0["minute"] = _d0.dateTime.dt.hour * 100 + _d0.dateTime.dt.minute
            _d0["tradeNotional"] = _d0.tradePrice * _d0.absFilledThisUpdate
            _d0 = \
                _d0.groupby(list(set(["minute", "secid"] + extra_grouping)))[[
                    "tradeNotional", "absFilledThisUpdate"
                ]].sum().reset_index()
            _d0["date"] = date
            _d0["tradeVwap"] = _d0.tradeNotional / _d0.absFilledThisUpdate

            #
            _d = pd.merge(
                _d,
                _d0,
                on = list(set(["date", "secid", "minute"] + extra_grouping)),
                how = "outer"
            )
            _d = pd.merge(
                _d,
                self._s_bar1m[_sec_name][self._s_bar1m[_sec_name].date == date],
                on = ["date", "secid", "minute"],
                how = "outer"
            )
            _col_to_fill = \
                ["orderNotional", "tradeNotional_final", "absOrderSize",
                 "tradeNotional", "absFilledThisUpdate"]
            for _col in _col_to_fill:
                _d[_col] = _d[_col].fillna(0.)

            _d["partTradePerStock"] = (_d.tradeNotional / _d.amount).clip(0., 1.)
            _d["improveOrderVwap"] = (_d.orderVwap / _d.vwap - 1.).clip(-1., 1.)
            _d["improveTradeVwap"] = (_d.tradeVwap / _d.vwap - 1.).clip(-1., 1.)

            _msum = _d.groupby(list(set(["date", "minute"] + extra_grouping)))[[
                "orderNotional", "tradeNotional_final", "tradeNotional",
                "amount"
            ]].sum().reset_index()
            _mean = _d.groupby(list(set(["date", "minute"] + extra_grouping)))[[
                "partTradePerStock", "improveOrderVwap", "improveTradeVwap",
                "sta", "sta_final", "mrm", "mrm_final"
            ]].mean().reset_index()

            _m = pd.merge(
                _msum,
                _mean,
                on = list(set(["date", "minute"] + extra_grouping)),
                how = "left"
            )
            _m["fillRate"] = _m.tradeNotional_final / _m.orderNotional
            _m["partTradePerMinute"] = _m.tradeNotional / _m.amount

            return _m

        except Exception as _e:
            print(_e)
            return pd.DataFrame()


    def _get_order_details_fast(
        self,
        date,
        extra_grouping = []
    ):
        '''
        A support function that derives the minutely order details,
        including only the conventional statistics without participation.
        No need to run "prepare" beforehand.

        Inputs
        ------
        date : int
        Date.

        extra_grouping : list of str, optional
        Needs to be column names of the order log.  Most commonly, "orderDirection".
        Default is an empty list [].
        '''
        _sec_name = utils.find_sec_name(date, self._ddict)
        if not _sec_name:
            return pd.DataFrame()

        try:
            kprefix = \
                self._rprefix\
                .format(_sec_name)

            _d = jdfs.read_file(
                os.path.join(kprefix, "orderLog_{}.csv{}".format(date, ".gz" if self._compressed else "")),
                self._pool,
                self._ns
            )
            _d0, _d = utils.process_order_log(_d)
            #
            _d["minute"] = _d.dateTime.dt.hour * 100 + _d.dateTime.dt.minute
            _d = \
                _d.groupby(["minute"] + extra_grouping)[[
                    "orderNotional", "tradeNotional_final", "absOrderSize"
                ]].sum().reset_index()
            _d["date"] = date

            #
            _d0 = _d0[_d0.updateType == 4]
            _d0["minute"] = _d0.dateTime.dt.hour * 100 + _d0.dateTime.dt.minute
            _d0["tradeNotional"] = _d0.tradePrice * _d0.absFilledThisUpdate
            _d0 = \
                _d0.groupby(["minute"] + extra_grouping)[[
                    "tradeNotional", "absFilledThisUpdate"
                ]].sum().reset_index()
            _d0["date"] = date
            #
            _d = pd.merge(
                _d,
                _d0,
                on = ["date", "minute"] + extra_grouping,
                how = "outer"
            )
            _col_to_fill = \
                ["orderNotional", "tradeNotional_final", "absOrderSize",
                 "tradeNotional", "absFilledThisUpdate"]
            for _col in _col_to_fill:
                _d[_col] = _d[_col].fillna(0.)

            _d["fillRate"] = _d.tradeNotional_final / _d.orderNotional

            return _d

        except Exception as _e:
            print(_e)
            return pd.DataFrame()


    def _get_inventory_details(
        self,
        date: int
    ):
        '''
        Support function to derive the daily inventory details.

        Inputs
        ------
        date : int
        Date.
        '''
        _sec_name = utils.find_sec_name(date, self._ddict)
        date_next = self._ra.dates.shift_trading_dates(date, 1)
        _sec_name_next = utils.find_sec_name(date_next, self._ddict_full)

        if (not _sec_name) or (not _sec_name_next):
            return pd.DataFrame()

        try:
            kprefix = self._rprefix.format(_sec_name)
            _readfunc = \
                (utils.read_raw_bytes_compressed if self._compressed else utils.read_raw_bytes)
            _t = _readfunc(
                os.path.join(kprefix, "fnLog_{}.csv{}".format(date, ".gz" if self._compressed else "")),
                kprefix.split("/")[1],
                kprefix.split("/")[2]
            )

            _d_summary = {"date": [date]}
            for _line in _t.readlines():
                if ("onSOD" in _line) and ("Start of day [{}]".format(date) in _line):
                    _d_summary["sod_cfe"] = \
                        [float(json.loads(_line.split("state: ")[-1].strip())["cashFreeEquity"])]
                    _d_summary["sod_mfe"] = \
                        [float(json.loads(_line.split("state: ")[-1].strip())["marginFreeEquity"])]
                elif ("onEOD" in _line) and ("End of day state" in _line):
                    _d_summary["eod_cfe"] = \
                        [float(json.loads(_line.split("state: ")[-1].strip())["cashFreeEquity"])]
                    _d_summary["eod_mfe"] = \
                        [float(json.loads(_line.split("state: ")[-1].strip())["marginFreeEquity"])]

            _v = jdfs.read_file(
                os.path.join(kprefix, "daiEnd_{}.csv{}".format(date, ".gz" if self._compressed else "")),
                kprefix.split("/")[1],
                kprefix.split("/")[2]
            )[["secid", "inv", "invS", "noStrategy"]]
            _v = pd.merge(
                _v,
                self._s_eod[_sec_name_next][self._s_eod[_sec_name_next].date == date_next],
                on = ["secid"],
                how = "left"
            )
            _v["mv"] = np.where(
                _v.inv > 0,
                _v.inv * _v.prvd_close_adj,
                0.0
            )
            _v["mvS"] = np.where(
                _v.invS > 0,
                _v.invS * _v.prvd_close_adj,
                0.0
            )
            _d_summary["eod_lmv"] = _v.mv.sum()
            _d_summary["eod_smv"] = _v.mvS.sum()

            # trading v holding
            _r = jdfs.read_file(
                os.path.join(kprefix, "dailySummary_{}.csv{}".format(date, ".gz" if self._compressed else "")),
                kprefix.split("/")[1],
                kprefix.split("/")[2],
            )[["secid", "inv_L", "inv_S"]]
            _r = pd.merge(
                _r,
                self._s_eod[_sec_name][self._s_eod[_sec_name].date == date],
                on = ["secid"],
                how = "left"
            )
            _r = pd.merge(
                _r,
                self._s_eod[_sec_name_next][self._s_eod[_sec_name_next].date == date_next],
                on = ["secid"],
                how = "left",
                suffixes = ["_y", "_t"]
            )
            _r["mv_holding_y"] = np.where(
                _r.inv_L > 0,
                _r.inv_L * _r.prvd_close_adj_y,
                0.0
            )
            _r["mv_holding_t"] = np.where(
                _r.inv_L > 0,
                _r.inv_L * _r.prvd_close_adj_t,
                0.0
            )
            _r["mvS_holding_y"] = np.where(
                _r.inv_S > 0,
                _r.inv_S * _r.prvd_close_adj_y,
                0.0
            )
            _r["mvS_holding_t"] = np.where(
                _r.inv_S > 0,
                _r.inv_S * _r.prvd_close_adj_t,
                0.0
            )
            _d_summary["sod_lmv_holding"] = _r.mv_holding_y.sum()
            _d_summary["eod_lmv_holding"] = _r.mv_holding_t.sum()
            _d_summary["sod_smv_holding"] = _r.mvS_holding_y.sum()
            _d_summary["eod_smv_holding"] = _r.mvS_holding_t.sum()

            _d_summary = pd.DataFrame(_d_summary)

            return _d_summary

        except Exception as _e:
            print(_e)
            return pd.DataFrame()

    def _get_inventory_details_perstock(
        self,
        date: int
    ):
        '''
        Support function to derive the daily inventory details.

        Inputs
        ------
        date : int
        Date.
        '''
        _sec_name = utils.find_sec_name(date, self._ddict)
        date_next = self._ra.dates.shift_trading_dates(date, 1)
        _sec_name_next = utils.find_sec_name(date_next, self._ddict_full)

        if (not _sec_name) or (not _sec_name_next):
            return pd.DataFrame()

        try:
            kprefix = self._rprefix.format(_sec_name)

            _v = jdfs.read_file(
                os.path.join(kprefix, "daiEnd_{}.csv{}".format(date, ".gz" if self._compressed else "")),
                kprefix.split("/")[1],
                kprefix.split("/")[2]
            )[["secid", "inv", "invS", "noStrategy"]]
            _v = pd.merge(
                _v,
                self._s_eod[_sec_name_next][self._s_eod[_sec_name_next].date == date_next],
                on = ["secid"],
                how = "left"
            )
            _v["mv_eod"] = np.where(
                _v.inv > 0,
                _v.inv * _v.prvd_close_adj,
                0.0
            )
            _v["mvS_eod"] = np.where(
                _v.invS > 0,
                _v.invS * _v.prvd_close_adj,
                0.0
            )
            _v["date"] = date
            _v["inv_eod"] = _v.inv.fillna(0).astype(int)
            _v["invS_eod"] = _v.invS.fillna(0).astype(int)
            _v["mv_eod"] = _v.mv_eod.fillna(0.0)
            _v["mvS_eod"] = _v.mvS_eod.fillna(0.0)
            _v = _v[["date", "secid", "inv_eod", "invS_eod", "mv_eod", "mvS_eod"]]

            # trading v holding
            _r = jdfs.read_file(
                os.path.join(kprefix, "dailySummary_{}.csv{}".format(date, ".gz" if self._compressed else "")),
                kprefix.split("/")[1],
                kprefix.split("/")[2],
            )[["secid", "inv_L", "inv_S"]]
            _r = pd.merge(
                _r,
                self._s_eod[_sec_name][self._s_eod[_sec_name].date == date],
                on = ["secid"],
                how = "left"
            )
            _r["mv_sod"] = np.where(
                _r.inv_L > 0,
                _r.inv_L * _r.prvd_close_adj,
                0.0
            )
            _r["mvS_sod"] = np.where(
                _r.inv_S > 0,
                _r.inv_S * _r.prvd_close_adj,
                0.0
            )
            _r = pd.merge(
                _r,
                self._s_eod[_sec_name_next][self._s_eod[_sec_name_next].date == date_next],
                on = ["secid"],
                how = "left",
                suffixes = ["", "_next"]
            )
            _r["mv_eod_nomove"] = np.where(
                _r.inv_L > 0,
                _r.inv_L * _r.prvd_close_adj_next,
                0.0
            )
            _r["mvS_eod_nomove"] = np.where(
                _r.inv_S > 0,
                _r.inv_S * _r.prvd_close_adj_next,
                0.0
            )
            _r["date"] = date
            _r["inv_sod"] = _r.inv_L.fillna(0).astype(int)
            _r["invS_sod"] = _r.inv_S.fillna(0).astype(int)
            _r["mv_sod"] = _r.mv_sod.fillna(0.0)
            _r["mvS_sod"] = _r.mvS_sod.fillna(0.0)
            _r["mv_eod_nomove"] = _r.mv_eod_nomove.fillna(0.0)
            _r["mvS_eod_nomove"] = _r.mvS_eod_nomove.fillna(0.0)
            
            _r = _r[[
                "date", "secid", "inv_sod", "invS_sod", "mv_sod", "mvS_sod",
                "mv_eod_nomove", "mvS_eod_nomove"
            ]]

            return pd.merge(_v, _r, on = ["date", "secid"], how = "outer")

        except Exception as _e:
            print(_e)
            return pd.DataFrame(), pd.DataFrame()

    def _get_po_exec_invexp_details(
        self,
        date : int,
    ):
        _sec_name = utils.find_sec_name(date, self._ddict)
        if not _sec_name:
            return pd.DataFrame(), pd.DataFrame()

        try:
            kprefix = self._rprefix.format(_sec_name)

            _po = jdfs.read_file(
                os.path.join(
                    kprefix,
                    "poMessageLog_{}.csv{}".format(date, ".gz" if self._compressed else "")
                ),
                self._pool,
                self._ns
            )
            _dai = jdfs.read_file(
                os.path.join(
                    kprefix,
                    "daiEnd_{}.csv{}".format(date, ".gz" if self._compressed else "")
                ),
                self._pool,
                self._ns
            )
            _dai["invL"] = _dai.inv
            _dai["invL_aft"] = _dai.inv
            _dai["invS_aft"] = _dai.invS
            _dai["price"] = _dai.lastPrice

            #
            _lastwave = pd.DataFrame({
                "secid": sorted(_po.secid.unique()),
            })
            _regular_minutes = sorted(_po.minuteCreated.unique())
            _lastwave["minuteCreated"] = _po.minuteCreated.max() + 1

            _lastwave = _lastwave.merge(
                _dai[[
                    "secid", "invL", "invL_aft", "invS", "invS_aft", "price"
                ]],
                on = ["secid"],
                how = "left",
            )
            _lastwave = _lastwave.fillna(0)

            _po = pd.concat([
                _po,
                _lastwave
            ])
            ###

            _dps = self._s_eod[_sec_name][["date", "jkey", "round_lot_size"]].copy()
            _dps = _dps[_dps.date == date].reset_index(drop = True)

            _po = _po.merge(
                _dps,
                left_on = ["secid"],
                right_on = ["jkey"],
                how = "left"
            )
            _po = _po[pd.notnull(_po.round_lot_size)]

            _po = _po.sort_values(by = ["secid", "minuteCreated"]).reset_index(drop = True)
            _po["invL_next"] = _po.groupby(["secid"]).invL.shift(-1)
            _po["invS_next"] = _po.groupby(["secid"]).invS.shift(-1)
            _po["price_next"] = _po.groupby(["secid"]).price.shift(-1)

            _po = _po[_po.minuteCreated.isin(_regular_minutes)]
            _po["inv_net"] = _po.invL - _po.invS
            _po["inv_net_aft"] = _po.invL_aft - _po.invS_aft
            _po["inv_net_aft_lsz"] = (_po.inv_net_aft // _po.round_lot_size * _po.round_lot_size).astype(int)
            _po["inv_net_next"] = _po.invL_next - _po.invS_next
            _po["move_aft"] = ((_po.inv_net_aft - _po.inv_net) // _po.round_lot_size * _po.round_lot_size).astype(int)
            _po["move_next"] = (_po.inv_net_next - _po.inv_net).astype(int)
            _po["long_target"] = (_po.move_aft > 0)
            _po["short_target"] = (_po.move_aft < 0)
            _po["long_moved"] = (_po.move_next > 0) & (_po.long_target)
            _po["short_moved"] = (_po.move_next < 0) & (_po.short_target)
            _po["mv_move_aft"] = _po.price * _po.move_aft
            _po["mv_move_next"] = _po.price * _po.move_next
            _po["fill_rate_long"] = np.where(
                _po.long_target,
                _po.move_next / _po.move_aft,
                0.
            )
            _po["fill_rate_short"] = np.where(
                _po.short_target,
                _po.move_next / _po.move_aft,
                0.
            )

            return _po

        except Exception as _e:
            print(_e)
            return pd.DataFrame()

    def _get_po_exec_details(
        self,
        date : int,
    ):
        _sec_name = utils.find_sec_name(date, self._ddict)
        if not _sec_name:
            return pd.DataFrame(), pd.DataFrame()

        try:
            kprefix = self._rprefix.format(_sec_name)

            _po = jdfs.read_file(
                os.path.join(
                    kprefix,
                    "poMessageLog_{}.csv{}".format(date, ".gz" if self._compressed else "")
                ),
                self._pool,
                self._ns
            )
            _dai = jdfs.read_file(
                os.path.join(
                    kprefix,
                    "daiEnd_{}.csv{}".format(date, ".gz" if self._compressed else "")
                ),
                self._pool,
                self._ns
            )
            _dai["invL"] = _dai.inv
            _dai["invL_aft"] = _dai.inv
            _dai["invS_aft"] = _dai.invS
            _dai["price"] = _dai.lastPrice

            #
            _lastwave = pd.DataFrame({
                "secid": sorted(_po.secid.unique()),
            })
            _regular_minutes = sorted(_po.minuteCreated.unique())
            _lastwave["minuteCreated"] = _po.minuteCreated.max() + 1

            _lastwave = _lastwave.merge(
                _dai[[
                    "secid", "invL", "invL_aft", "invS", "invS_aft", "price"
                ]],
                on = ["secid"],
                how = "left",
            )
            _lastwave = _lastwave.fillna(0)

            _po = pd.concat([
                _po,
                _lastwave
            ])
            ###

            _dps = self._s_eod[_sec_name][["date", "jkey", "round_lot_size"]].copy()
            _dps = _dps[_dps.date == date].reset_index(drop = True)

            _po = _po.merge(
                _dps,
                left_on = ["secid"],
                right_on = ["jkey"],
                how = "left"
            )
            _po = _po[pd.notnull(_po.round_lot_size)]

            _po = _po.sort_values(by = ["secid", "minuteCreated"]).reset_index(drop = True)
            _po["invL_next"] = _po.groupby(["secid"]).invL.shift(-1)
            _po["invS_next"] = _po.groupby(["secid"]).invS.shift(-1)
            _po["price_next"] = _po.groupby(["secid"]).price.shift(-1)

            _po = _po[_po.minuteCreated.isin(_regular_minutes)]
            _po["inv_net"] = _po.invL - _po.invS
            _po["inv_net_aft"] = _po.invL_aft - _po.invS_aft
            _po["inv_net_aft_lsz"] = (_po.inv_net_aft // _po.round_lot_size * _po.round_lot_size).astype(int)
            _po["inv_net_next"] = _po.invL_next - _po.invS_next
            _po["move_aft"] = ((_po.inv_net_aft - _po.inv_net) // _po.round_lot_size * _po.round_lot_size).astype(int)
            _po["move_next"] = (_po.inv_net_next - _po.inv_net).astype(int)
            _po["long_target"] = (_po.move_aft > 0)
            _po["short_target"] = (_po.move_aft < 0)
            _po["ssell_target"] = (_po.move_aft < 0) & \
                (_po.inv_net > 0)
            _po["sshort_target"] = (_po.move_aft < 0) & \
                ((_po.inv_net <= 0) | ((_po.inv_net > 0) & (abs(_po.move_aft) > _po.inv_net)))
            _po["long_moved"] = (_po.move_next > 0) & (_po.long_target)
            _po["short_moved"] = (_po.move_next < 0) & (_po.short_target)
            _po["ssell_moved"] = (_po.move_next < 0) & (_po.ssell_target)
            _po["sshort_moved"] = (_po.move_next < 0) & (_po.sshort_target)
            _po["mv_move_aft"] = _po.price * _po.move_aft
            _po["mv_move_next"] = _po.price * _po.move_next
            _po["move_aft_ssell"] = np.where(
                _po.ssell_target,
                -abs(_po[["move_aft", "inv_net"]]).min(axis = 1),
                0
            )
            _po["move_aft_sshort"] = np.where(
                _po.inv_net <= 0,
                _po.move_aft,
                -(abs(_po.move_aft) - abs(_po.inv_net)).clip(0, None)
            )
            _po["move_aft_sshort"] = np.where(
                _po.sshort_target,
                _po.move_aft_sshort,
                0
            )
            _po["move_next_ssell"] = np.where(
                _po.ssell_target,
                -abs(_po[["move_next", "inv_net"]]).min(axis = 1),
                0
            )
            _po["move_next_sshort"] = np.where(
                _po.inv_net <= 0,
                _po.move_next,
                -(abs(_po.move_next) - abs(_po.inv_net)).clip(0, None)
            )
            _po["move_next_sshort"] = np.where(
                _po.sshort_target,
                _po.move_next_sshort,
                0
            )
            _po["mv_move_aft_ssell"] = _po.price * _po.move_aft_ssell
            _po["mv_move_aft_sshort"] = _po.price * _po.move_aft_sshort
            _po["mv_move_next_ssell"] = _po.price * _po.move_next_ssell
            _po["mv_move_next_sshort"] = _po.price * _po.move_next_sshort
            _po["fill_rate_long"] = np.where(
                _po.long_target,
                _po.move_next / _po.move_aft,
                0.
            )
            _po["fill_rate_short"] = np.where(
                _po.short_target,
                _po.move_next / _po.move_aft,
                0.
            )
            _po["fill_rate_ssell"] = np.where(
                _po.ssell_target,
                _po.move_next_ssell / _po.move_aft_ssell,
                0.
            )
            _po["fill_rate_sshort"] = np.where(
                _po.sshort_target,
                _po.move_next_sshort / _po.move_aft_sshort,
                0.
            )

            _fill_rate_intraday = utils._ifr_calculator(_po)

            _po_min = pd.DataFrame({
                "minuteCreated": sorted(_po.minuteCreated.unique()),
            })
            for _feat in ["mv_move_aft", "mv_move_next"]:
                for _dir in ["long_target", "short_target", "ssell_target", "sshort_target"]:
                    _colname = "{}_{}".format(_feat, _dir.split("_")[0])
                    _df = _po[_po[_dir]].groupby(["minuteCreated"])[
                        (_colname if _dir in ["ssell_target", "sshort_target"] else _feat)
                    ].sum().reset_index()
                    _df[_colname] = _df[
                        (_colname if _dir in ["ssell_target", "sshort_target"] else _feat)
                    ]

                    _po_min = _po_min.merge(
                        _df[["minuteCreated", _colname]],
                        on = ["minuteCreated"],
                        how = "left"
                    )

            _po_min["min_fill_rate_long"] = _po_min.mv_move_next_long / _po_min.mv_move_aft_long
            _po_min["min_fill_rate_short"] = _po_min.mv_move_next_short / _po_min.mv_move_aft_short
            _po_min["min_fill_rate_ssell"] = _po_min.mv_move_next_ssell / _po_min.mv_move_aft_ssell
            _po_min["min_fill_rate_sshort"] = _po_min.mv_move_next_sshort / _po_min.mv_move_aft_sshort

            _po_min_pct = pd.merge(
                _po[_po.long_target].groupby(["minuteCreated"]).fill_rate_long.describe(
                    percentiles = []
                ).reset_index()[[
                    "minuteCreated", "count", "mean"
                ]],
                _po[_po.short_target].groupby(["minuteCreated"]).fill_rate_short.describe(
                    percentiles = []
                ).reset_index()[[
                    "minuteCreated", "count", "mean"
                ]],
                on = ["minuteCreated"],
                how = "outer",
                suffixes = ["_long", "_short"]
            ).fillna(0.)
            _po_min_count = _po.groupby(["minuteCreated"])[[
                "long_moved", "short_moved"
            ]].sum().reset_index()
            _po_min_pct = _po_min_pct.merge(
                _po_min_count,
                on = ["minuteCreated"],
                how = "left"
            )

            _fill_rate_long = _po_min.mv_move_next_long.sum() / _po_min.mv_move_aft_long.sum()
            _fill_rate_short = _po_min.mv_move_next_short.sum() / _po_min.mv_move_aft_short.sum()
            _fill_rate_ssell = _po_min.mv_move_next_ssell.sum() / _po_min.mv_move_aft_ssell.sum()
            _fill_rate_sshort = _po_min.mv_move_next_sshort.sum() / _po_min.mv_move_aft_sshort.sum()

            # modify here when doing sta capture
            _colkeep = ["minuteCreated"] + [
                "mv_move_aft_long",
                "mv_move_next_long",
                "mv_move_aft_short",
                "mv_move_next_short",
                "min_fill_rate_long",
                "min_fill_rate_short",
                "min_fill_rate_ssell",
                "min_fill_rate_sshort"
            ]
            _po_min = _po_min[_colkeep].merge(
                _po_min_pct,
                on = ["minuteCreated"],
                how = "left"
            )
            _po_min["date"] = date

            _po_daily = pd.DataFrame({
                "date": [date],
                "fill_rate_intraday": [_fill_rate_intraday],
                "fill_rate_long": [_fill_rate_long],
                "fill_rate_short": [_fill_rate_short],
                "fill_rate_ssell": [_fill_rate_ssell],
                "fill_rate_sshort": [_fill_rate_sshort]
            })

            return _po_min, _po_daily

        except Exception as _e:
            print(_e)
            return pd.DataFrame(), pd.DataFrame()

    def _get_barra_exposure_details(
        self,
        date,
    ):
        _sec_name = utils.find_sec_name(date, self._ddict)
        if not _sec_name:
            return pd.DataFrame()

        try:
            kprefix = self._rprefix.format(_sec_name)
            _eod = self._s_eod[_sec_name].copy()
            _beta = self._s_beta[_sec_name].copy()

            _exp = utils.process_barra_exposure(
                date,
                self._barra_model,
                mta_nas = self._mta_nas
            )

            _dai = jdfs.read_file(
                os.path.join(
                    kprefix,
                    "daiEnd_{}.csv{}".format(date, ".gz" if self._compressed else "")
                ),
                self._pool,
                self._ns,
            )
            _dai["net_inv"] = _dai.inv - _dai.invS

            _dai = _dai[["secid", "inv", "invS", "net_inv"]]
            _dai = pd.merge(
                _dai,
                _eod[_eod.date == date],
                on = ["secid"],
                how = "left"
            )
            _dai["exposure_mv"] = _dai.net_inv * _dai["close"]
            _dai["exposure_wgt"] = _dai.exposure_mv / self._cap

            # use beta from ms_mkt_beta
            _df_beta = pd.merge(
                _dai,
                _beta[_beta.date == date][["secid", "beta_{}".format(self._bm)]],
                on = ["secid"],
                how = "left",
            )
            _mkt_beta = (
                _df_beta.exposure_wgt * _df_beta["beta_{}".format(self._bm)]
            ).sum()

            # exposure
            _df = pd.merge(_dai, _exp[["secid", "Factor", "Exposure"]],
                           on = ["secid"],
                           how = "left")
            _df["wgt_exp"] = _df.exposure_wgt * _df.Exposure
            _df_out = _df.groupby(["Factor"])[["wgt_exp"]].sum().reset_index()
            _df_out["date"] = date

            _df_out = _df_out.T
            _df_out.columns = _df_out.loc["Factor"]
            _df_out.drop(_df_out.index[[0, 2]], inplace = True)

            _df_out["date"] = date
            _df_out["mkt_beta"] = _mkt_beta

            return _df_out

        except Exception as _e:
            print(_e)
            return None

    def _get_barra_exposure_details_intraday(
        self,
        date,
    ):
        _sec_name = utils.find_sec_name(date, self._ddict)
        if not _sec_name:
            return pd.DataFrame()

        try:
            kprefix = self._rprefix.format(_sec_name)
            _eod = self._s_eod[_sec_name].copy()
            _beta = self._s_beta[_sec_name].copy()

            _exp = utils.process_barra_exposure(
                date,
                self._barra_model,
                mta_nas = self._mta_nas
            )

            _dai = jdfs.read_file(
                os.path.join(
                    kprefix,
                    "daiEnd_{}.csv{}".format(date, ".gz" if self._compressed else "")
                ),
                self._pool,
                self._ns,
            )
            _dai["net_inv"] = _dai.inv - _dai.invS

            _dai = _dai[["secid", "inv", "invS", "net_inv"]]
            _dai = pd.merge(
                _dai,
                _eod[_eod.date == date],
                on = ["secid"],
                how = "left"
            )
            _dai["exposure_mv"] = _dai.net_inv * _dai["close"]
            _dai["exposure_wgt"] = _dai.exposure_mv / self._cap

            # use beta from ms_mkt_beta
            _df_beta = pd.merge(
                _dai,
                _beta[_beta.date == date][["secid", "beta_{}".format(self._bm)]],
                on = ["secid"],
                how = "left",
            )
            _mkt_beta = (
                _df_beta.exposure_wgt * _df_beta["beta_{}".format(self._bm)]
            ).sum()

            # exposure
            _df = pd.merge(_dai, _exp[["secid", "Factor", "Exposure"]],
                           on = ["secid"],
                           how = "left")
            _df["wgt_exp"] = _df.exposure_wgt * _df.Exposure
            _df_out = _df.groupby(["Factor"])[["wgt_exp"]].sum().reset_index()
            _df_out["date"] = date

            _df_out = _df_out.T
            _df_out.columns = _df_out.loc["Factor"]
            _df_out.drop(_df_out.index[[0, 2]], inplace = True)

            _df_out["date"] = date
            _df_out["mkt_beta"] = _mkt_beta

            return _df_out

        except Exception as _e:
            print(_e)
            return None


def _process_inventory_private(
    _input_lighter,
    force = False
):

    if force or (_input_lighter._d_inventory is None):
        logging.warning(
            "[_process_inventory_private] Processing of inventory/cash details; estimated to take %.1f minutes.",
            0.027 * len(_input_lighter._date_list) / 60
        )

        time0 = time.time()

        def _func_to_map(_date):
            _dfi = _input_lighter._get_inventory_details(_date)
            return _dfi

        with multiprocessing.Pool(8) as P:
            _l_res = P.map(_func_to_map, _input_lighter._date_list)
            P.close()
            P.join()
        
        _df = pd.concat(_l_res)
        _df = _df.sort_values(
            by = ["date"]
        ).reset_index(drop = True)

        _input_lighter._d_inventory = _df

        logging.info(
            "[_process_inventory_private] %i of %i days loaded; it took %.2f seconds.",
            len(_input_lighter._d_inventory.date.unique()),
            len(_input_lighter._date_list),
            time.time() - time0
        )

    else:
        logging.warning(
            "[_process_inventory_private] Skip because the table exists; use force = True to force a reload."
        )
        
    return _input_lighter
