#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
import platform
import sys
import argparse

import csv
import json
import math

import time
from datetime import datetime, timedelta

import pandas as pd
import requests
from pandas import Series, DataFrame
from pvlib import irradiance, solarposition


class WRAPI_merra2:

    def __init__(self, ddir='', encoding='utf-8'):
        self.debug = False

        if '' == ddir:
            ddir, filename = os.path.split(os.path.abspath(sys.argv[0]))

        if (platform.system() in ["Windows"]):
            ddir += '\\'
        else:
            ddir += '/'

        conf_file = ddir + 'config.json'
        access_file = ddir + 'key'

        self.access = str()
        with open(access_file, mode='r', encoding=encoding) as f:
            try:
                self.access = str(f.read())
            except Exception as e:
                f = sys.exc_info()[2].tb_frame.f_back
                print("error: %s, line %s, in %s" % (f.f_code.co_filename,
                                                     str(f.f_lineno), f.f_code.co_name))
                print(e)

        self.headers_url = {'access-token': self.access}

        conf = {
            "base_url": "http://api.goldwind.com.cn/",
            "merra2_download": "merra2Download",
            "timezone": 8,
            "line_offset": 12,
            "merra2_select": "selectFourPoints/merra2data",
            "merra2_download_params": "SWGNT,PS,T10M,U10M,V10M",
            "start_date": "1980-01-01",
            "end_date": "2017-12-31",
            "min_lon": -180.0,
            "max_lon": 179.375,
            "min_lat": -90.0,
            "max_lat": 90.0,
            "timeout": 300
        }
        self.col_name = dict()
        self.units = dict()
        self.solar_dir = ddir + 'solar'
        with open(conf_file, mode='r', encoding=encoding) as f:
            try:
                info = json.load(f)
                conf = info['WRAPI']
                self.col_name = info['headers']
                self.units = info['units']
                if info['solar_dir']:
                    self.solar_dir = info['solar_dir']
            except Exception as e:
                f = sys.exc_info()[2].tb_frame.f_back
                print("error: %s, line %s, in %s" % (f.f_code.co_filename,
                                                     str(f.f_lineno), f.f_code.co_name))
                print(e)

        self.download_url = str(conf['base_url']) + str(conf['merra2_download'])
        self.download_params = str(conf['merra2_download_params'])
        self.line_offset = int(conf['line_offset'])
        self.select_url = str(conf['base_url']) + str(conf['merra2_select'])
        self.start_date = datetime.strptime(conf['start_date'], '%Y-%m-%d')
        self.end_date = datetime.strptime(conf['end_date'], '%Y-%m-%d')
        self.min_lon = float(conf['min_lon'])
        self.max_lon = float(conf['max_lon'])
        self.min_lat = float(conf['min_lat'])
        self.max_lat = float(conf['max_lat'])
        self.timeout = int(conf['timeout'])
        self.retry = 0

        # 筛选数量满足的月份
        self.days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        self.timezone = 8
        self.debug_level = 0

    def solardir(self):
        return (self.solar_dir)

    def set_debug(self, debug=True, level=1):
        self.debug = debug
        self.debug_level = level
        return

    def check_debug(self, level=1):
        result = bool(self.debug) and (self.debug_level >= level)
        return result

    def days_in_month(self, month):
        return self.days[month - 1]

    def tz(self):
        return self.timezone

    def colname(self):
        return list(self.col_name.keys())

    def restart_con(self, count=3):
        self.retry = 3
        return

    def check_con(self):
        ok = (self.retry > 0)
        if (ok):
            self.retry -= 1
        return ok

    def shutdown_con(self):
        self.retry = 0
        return

    def strptime(self, time):
        return datetime.strptime(time, '%Y-%m-%dT%H:%M:%S')

    def strftime(self, time):
        return datetime.strptime(time, '%Y-%m-%dT%H:%M:%S').strftime('%Y-%m-%d %H:%M:%S')

    def timeindex(self, time):
        dt = strptime(time)
        return pd.to_datetime(dt)

    def get_attri(self):
        return (self)

    # 读数据, '20180101'
    def read(self, lat, lon, start, end):
        payload = {
            'lat': lat,
            'lon': lon,
            'start_time': start,
            'end_time': end,
            'parameter_data': self.download_params
        }
        if self.check_debug(2):
            print("payload: ")
            print(payload)

        raw = DataFrame(columns=self.colname())

        try:
            r = requests.post(self.download_url, params=payload,
                              headers=self.headers_url,
                              timeout=self.timeout)
            rows = r.text.splitlines()
            # 产生dataframe
            head = rows[self.line_offset].split('\t')
            val = list(self.col_name.values())
            key = list(self.col_name.keys())
            # 待优化，向量化操作
            data = []
            for row in rows[(self.line_offset + 1):]:
                data.append(row.split('\t'))

            raw = DataFrame(data, columns=head)
            # 删除多余列
            for i in range(len(head)):
                if head[i] in val:  # 改名
                    raw.rename(columns={head[i]: key[val.index(head[i])]},
                               inplace=True)
                else:  # 删除
                    raw.drop([head[i]], axis=1, inplace=True)
        # 待优化异常处理
        except requests.exceptions.Timeout:
            print("error: 请求数据超时, 经度= %f, 维度= %f, 日期: %s - %s"
                  % (float(lon), float(lat), str(start), str(end)))
        except Exception as e:
            f = sys.exc_info()[2].tb_frame.f_back
            print("error: %s, line %s, in %s" % (f.f_code.co_filename,
                                                 str(f.f_lineno), f.f_code.co_name))
            print(e)
            print(r.text)

        return (raw)

    # '2017-01-01'
    def read_date(self, lat, lon, date):
        if self.check_debug():
            print("getting: 日期: %s" % date)

        self.restart_con()
        df = DataFrame(columns=self.colname())
        try:
            lon = float(lon)
            lat = float(lat)
            dt = datetime.strptime(date, '%Y-%m-%d')
            start = dt.strftime('%Y%m%d')
            end = dt.strftime('%Y%m%d')
            while self.check_con():
                raw = self.read(lat, lon, start, end)
                if not raw.empty:
                    df = pd.concat([df, raw], ignore_index=True)[self.colname()]
                    break
                else:
                    time.sleep(2)
        except Exception as e:
            f = sys.exc_info()[2].tb_frame.f_back
            print("error: %s, line %s, in %s" % (f.f_code.co_filename,
                                                 str(f.f_lineno), f.f_code.co_name))
            print(e)
            self.shutdown_con()

        if df.empty and self.check_debug():
            print("missing: %s" % date)

        return (df)

    # '2017-01'
    def readmulti_month(self, lat, lon, month):
        df = DataFrame(columns=self.colname())
        try:
            lon = float(lon)
            lat = float(lat)
            start = datetime.strptime(month, '%Y-%m')
            for i in range(self.days_in_month(start.month)):
                dt = start + timedelta(days=i)
                raw = self.read_date(lat, lon, dt.strftime('%Y-%m-%d'))
                if not raw.empty:
                    df = pd.concat([df, raw], ignore_index=True)
                else:
                    if self.check_debug():
                        print("Incomplete data: %s" % month)
                    break
        except Exception as e:
            f = sys.exc_info()[2].tb_frame.f_back
            print("error: %s, line %s, in %s" % (f.f_code.co_filename,
                                                 str(f.f_lineno), f.f_code.co_name))
            print(e)

        if df.empty and self.check_debug():
            print("missing: %s" % month)

        return (df)

    # '2017-01'
    def read_month(self, lat, lon, month):
        if self.check_debug():
            print("getting: 月份: %s" % month)

        self.restart_con()
        df = DataFrame(columns=self.colname())

        try:
            lon = float(lon)
            lat = float(lat)
            start = datetime.strptime(month, '%Y-%m')
            days = self.days_in_month(start.month)-1
            end = start + timedelta(days=days)
            start = start.strftime('%Y%m%d')
            end = end.strftime('%Y%m%d')
            while self.check_con():
                raw = self.read(lat, lon, start, end)
                if not raw.empty:
                    df = pd.concat([df, raw], ignore_index=True)[self.colname()]
                    df.sort_values(by=["datetime"], ascending=True, inplace=True)
                    break
                else:
                    time.sleep(2)
        except Exception as e:
            f = sys.exc_info()[2].tb_frame.f_back
            print("error: %s, line %s, in %s" % (f.f_code.co_filename,
                                                 str(f.f_lineno), f.f_code.co_name))
            print(e)
            self.shutdown_con()

        if df.empty and self.check_debug():
            print("missing: %s" % month)

        return (df)

        # '2017'
    def readmulti_year(self, lat, lon, year):
        df = DataFrame(columns=self.colname())
        try:
            lon = float(lon)
            lat = float(lat)
            start = datetime.strptime(year, '%Y')
            for i in range(1, 13):
                dt = datetime(start.year, i, 1)
                raw = self.read_month(lat, lon, dt.strftime('%Y-%m'))
                if not raw.empty:
                    df = pd.concat([df, raw], ignore_index=True)
                else:
                    if self.check_debug():
                        print("Incomplete data: %s" % year)
                    break
        except Exception as e:
            f = sys.exc_info()[2].tb_frame.f_back
            print("error: %s, line %s, in %s" % (f.f_code.co_filename,
                                                 str(f.f_lineno), f.f_code.co_name))
            print(e)

        if df.empty and self.check_debug():
            print("missing: %s" % year)

        return (df)

    def read_year(self, lat, lon, year):
        if self.check_debug():
            print("getting: 年份: %s" % year)

        self.restart_con()
        df = DataFrame(columns=self.colname())
        try:
            lon = float(lon)
            lat = float(lat)
            start = datetime.strptime(year, '%Y')
            end = datetime(start.year, 12, 31)
            start = start.strftime('%Y%m%d')
            end = end.strftime('%Y%m%d')

            while self.check_con():
                raw = self.read(lat, lon, start, end)
                if not raw.empty:
                    # 删除闰年2月29日
                    index = pd.DatetimeIndex(raw.datetime)
                    spec = (index.month == 2) & (index.day > 28)
                    raw.drop(raw.index[spec], axis=0, inplace=True)
                    df = pd.concat([df, raw], ignore_index=True)[self.colname()]
                    df.sort_values(by=["datetime"], ascending=True, inplace=True)
                    break
                else:
                    time.sleep(2)
        except Exception as e:
            f = sys.exc_info()[2].tb_frame.f_back
            print("error: %s, line %s, in %s" % (f.f_code.co_filename,
                                                 str(f.f_lineno), f.f_code.co_name))
            print(e)
            self.shutdown_con()

        if df.empty and self.check_debug():
            print("missing: %s" % year)

        return (df)

    # time  is DatetimeIndex
    def zenith(self, time, lat, lon):
        zenith = solarposition.get_solarposition(time, lat, lon).zenith.values
        return (zenith)

    def ghi(self, swgnt):
        ghi = []
        swgnt = Series(swgnt)
        f = lambda x: round(float(x), 2)
        try:
            if self.units['SWGNT'] in ['w/m2']:
                ghi = swgnt.apply(f).values
        except Exception as e:
            f = sys.exc_info()[2].tb_frame.f_back
            print("error: %s, line %s, in %s" % (f.f_code.co_filename,
                                                 str(f.f_lineno), f.f_code.co_name))
            print(e)
        return (ghi)

    def temperature(self, temperature, src='K', dst='C'):
        temp = []
        temperature = Series(temperature)
        f = {'C': self.temperature_C, 'c': self.temperature_C}
        try:
            temp = f[dst](temperature, src)
        except Exception as e:
            f = sys.exc_info()[2].tb_frame.f_back
            print("error: %s, line %s, in %s" % (f.f_code.co_filename,
                                                 str(f.f_lineno), f.f_code.co_name))
            print(e)
        return (temp)

    def temperature_C(self, temperature, unit='K'):
        temp = []
        try:
            if unit in ['K']:
                f = lambda x: round((float(x) - 273.15), 2)
                temp = temperature.apply(f).values
            elif unit in ['C']:
                f = lambda x: round(float(x), 2)
                temp = temperature.apply(f).values
        except Exception as e:
            f = sys.exc_info()[2].tb_frame.f_back
            print("error: %s, line %s, in %s" % (f.f_code.co_filename,
                                                 str(f.f_lineno), f.f_code.co_name))
            print(e)

        return (temp)

    def pressure(self, pressure, unit='mbar'):
        ps = []
        pressure = Series(pressure)
        try:
            if self.units['PS'] in unit:
                coef = 1.0
            elif (self.units['PS'] in ['Pa']) and (unit in ['mbar', 'mb']):
                coef = 1.0 / 100.0
            else:  # (self.units['PS'] in ['mbar', 'mb']) and (unit in ['Pa']):
                coef = 100.0

            f = lambda x: round((float(x) * coef), 2)
            ps = pressure.apply(f).values
        except Exception as e:
            f = sys.exc_info()[2].tb_frame.f_back
            print("error: %s, line %s, in %s" % (f.f_code.co_filename,
                                                 str(f.f_lineno), f.f_code.co_name))
            print(e)

        return (ps)

    # time  is DatetimeIndex, pressure is Pa
    def dni(self, time, lat, lon, ghi, pressure):
        dni = []
        try:
            ghi = self.ghi(ghi)
            ps = self.pressure(pressure, 'Pa')
            zenith = self.zenith(time, lat, lon)
            dni = irradiance.disc(ghi, zenith, time, ps).dni.values
            # irradiance.dirint(ghi, zenith, time, ps)
        except Exception as e:
            f = sys.exc_info()[2].tb_frame.f_back
            print("error: %s, line %s, in %s" % (f.f_code.co_filename,
                                                 str(f.f_lineno), f.f_code.co_name))
            print(e)

        return (dni)

    # time  is DatetimeIndex
    def dhi(self, time, lat, lon, ghi):
        dhi = []
        try:
            ghi = self.ghi(ghi)
            zenith = self.zenith(time, lat, lon)
            dhi = irradiance.erbs(ghi, zenith, time)['dhi']
        except Exception as e:
            f = sys.exc_info()[2].tb_frame.f_back
            print("error: %s, line %s, in %s" % (f.f_code.co_filename,
                                                 str(f.f_lineno), f.f_code.co_name))
            print(e)

        return (dhi)

    def uv2dirn(self, df):
        vm = float(df.vm)
        um = float(df.um)
        if abs(vm) < 0.000001:  # No v-component of velocity
            if vm >= 0:
                dm = 270.0
            else:
                dm = 90.0
        else:  # Calculate angle and convert to degrees
            theta = math.atan(um / vm)
            theta = math.degrees(theta)
            if vm > 0:
                dm = int(theta + 180.0)
            else:  # Make sure angle is positive
                theta = theta + 360.0
                dm = int(theta % 360.0)
        return dm

    def wind_dirn(self, um, vm):
        dirn = []
        try:
            df = DataFrame(data={'um': um, 'vm': vm},
                           columns=['um', 'vm'])
            dirn = df.apply(self.uv2dirn, axis='columns').values
        except Exception as e:
            f = sys.exc_info()[2].tb_frame.f_back
            print("error: %s, line %s, in %s" % (f.f_code.co_filename,
                                                 str(f.f_lineno), f.f_code.co_name))
            print(e)

        return (dirn)

    def uv2speed(self, df):
        vm = float(df.vm)
        um = float(df.um)
        sm = round(math.sqrt(um * um + vm * vm), 4)
        return sm

    def wind_speed(self, um, vm):
        sp = []
        try:
            df = DataFrame(data={'um': um, 'vm': vm},
                           columns=['um', 'vm'])
            sp = df.apply(self.uv2speed, axis='columns').values
        except Exception as e:
            f = sys.exc_info()[2].tb_frame.f_back
            print("error: %s, line %s, in %s" % (f.f_code.co_filename,
                                                 str(f.f_lineno), f.f_code.co_name))
            print(e)

        return (sp)

    def read_grid(self, lat, lon):
        payload = {
            'lat': lat,
            'lon': lon
        }
        self.restart_con()
        while self.check_con():
            try:
                r = requests.get(self.select_url, params=payload,
                                 headers=self.headers_url,
                                 timeout=self.timeout)
                lon, lat = r.json()["four_points_dis"][0][0]
            # 待优化异常处理
            except requests.exceptions.Timeout:
                print("error: 请求坐标超时, 经度= %f, 维度= %f"
                      % (float(lon), float(lat)))
            # except Exception as e:
            except Exception as e:
                f = sys.exc_info()[2].tb_frame.f_back
                print("error: %s, line %s, in %s" % (f.f_code.co_filename,
                                                     str(f.f_lineno), f.f_code.co_name))
                print(e)
                time.sleep(2)
            else:
                break

        alti = 0
        return lat, lon, alti

    def sam_filename(self, lat, lon, date):
        filename = 'solar' + '_' + str(lat) + '_' + str(lon) \
                   + '_' + date + '.csv'
        if (platform.system() in ["Windows"]):
            out = self.solardir() + '\\' + filename
        else:
            out = self.solardir() + '/' + filename
        return out

    def sam_head(self, out, lat, lon, alti, tz=8):
        with open(out, 'w', encoding='utf-8', newline='') as f:
            cf = csv.writer(f)
            cf.writerow(['Source', 'Location ID', 'City', 'State', 'Country',
                         'Latitude', 'Longitude', 'Time Zone', 'Elevation',
                         'Local Time Zone', 'Dew Point Units', 'DHI Units',
                         'DNI Units', 'GHI Units', 'Temperature Units',
                         'Pressure Units', 'Wind Direction Units', 'Wind Speed',
                         'Version'])
            cf.writerow(['-', '-', '-', '-', '-',
                         lat, lon, self.tz(), alti, tz,
                         'c', 'w/m2', 'w/m2', 'w/m2', 'c', 'mbar',
                         'Degrees', 'm/s', 'v0.0.0'])

        return

    # 先不考虑：时区问题，以及时区引起的日期改变，尤其二月的日期改变
    def sam_tmy(self, out, lat, lon, raw, tz=8, header=0):
        lat = round(float(lat), 4)
        lon = round(float(lon), 4)

        time = pd.DatetimeIndex(raw.datetime)
        cols = ['Year', 'Month', 'Day', 'Hour', 'Minute', # 'Dew Point',
                'DHI', 'DNI', 'GHI', 'Pressure',
                'Temperature', 'Wind Direction', 'Wind Speed']
        tmy = DataFrame(data={
            'Year': time.year,
            'Month': time.month,
            'Day': time.day,
            'Hour': time.hour,
            'Minute': time.minute,
           # 'Dew Point': Series(None),
            'DHI': Series(self.dhi(time, lat, lon, raw.SWGNT)),
            'DNI': Series(self.dni(time, lat, lon, raw.SWGNT, raw.PS)),
            'GHI': Series(self.ghi(raw.SWGNT)),
            'Pressure': Series(self.pressure(raw.PS, unit='mbar')),  # mbar
            'Temperature': Series(self.temperature(raw.T10M, 'K', 'C')),  # C
            'Wind Direction': Series(self.wind_dirn(raw.U10M, raw.V10M)),
            'Wind Speed': Series(self.wind_speed(raw.U10M, raw.V10M))},
            columns=cols)

        tmy.to_csv(out, index=False, sep=',', mode='a',
                   encoding='utf-8', line_terminator='\n', header=header)

        if self.check_debug():
            print("finished %s" % out)
        return

    def sam_year_mutli(self, lat, lon, year='2017', tz=8):
        dt = datetime.strptime(year, '%Y')
        grid_lat, grid_lon, grid_alti = self.read_grid(lat, lon)
        # 切换到栅格数据坐标
        month = datetime(dt.year, dt.month, 1).strftime('%Y-%m')
        raw = self.read_month(grid_lat, grid_lon, month)
        if not raw.empty:
            out = self.sam_filename(lat, lon, dt.strftime('%Y'))
            self.sam_head(out, grid_lat, grid_lon, grid_alti, tz)
            self.sam_tmy(out, grid_lat, grid_lon, raw, tz, header=True)
            for i in range(2, 13):
                time.sleep(2)
                month = datetime(dt.year, i, 1).strftime('%Y-%m')
                raw = self.read_month(grid_lat, grid_lon, month)
                if not raw.empty:
                    self.sam_tmy(out, grid_lat, grid_lon, raw, tz)
        return

    def sam_year(self, lat, lon, year='2017', tz=8):
        dt = datetime.strptime(year, '%Y')
        grid_lat, grid_lon, grid_alti = self.read_grid(lat, lon)
        # 切换到栅格数据坐标
        raw = self.read_year(grid_lat, grid_lon, year)
        if not raw.empty:
            out = self.sam_filename(lat, lon, dt.strftime('%Y'))
            self.sam_head(out, grid_lat, grid_lon, grid_alti, tz)
            self.sam_tmy(out, grid_lat, grid_lon, raw, tz, header=True)
        return

    def sam_month(self, lat, lon, month='2017-01', tz=8):
        dt = datetime.strptime(month, '%Y-%m')
        grid_lat, grid_lon, grid_alti = self.read_grid(lat, lon)
        # 切换到栅格数据坐标
        raw = self.read_month(grid_lat, grid_lon, month)
        if not raw.empty:
            out = self.sam_filename(lat, lon, dt.strftime('%Y%m'))
            self.sam_head(out, grid_lat, grid_lon, grid_alti, tz)
            self.sam_tmy(out, grid_lat, grid_lon, raw, tz, header=True)
        return

    def sam_date(self, lat, lon, date='2017-01-01', tz=8):
        dt = datetime.strptime(date, '%Y-%m-%d')
        grid_lat, grid_lon, grid_alti = self.read_grid(lat, lon)
        raw = self.read_date(lat, lon, date)
        if not raw.empty:
            out = self.sam_filename(lat, lon, dt.strftime('%Y%m%d'))
            self.sam_head(out, grid_lat, grid_lon, grid_alti, tz)
            self.sam_tmy(out, grid_lat, grid_lon, raw, tz, header=True)
        return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate a Special Meteorological Year')
    parser.add_argument('coord', nargs='*', help='coordinate: lat lon')
    parser.add_argument('-y', '--year', help='Download special year, eg: 2017')
    parser.add_argument('-a', '--auto', help='Download all special meteorological year in China, eg: 2017')
    parser.add_argument('-m', '--month', help='Download special month, eg: 2017-01')
    parser.add_argument('-d', '--date', help='Download special date, eg: 2017-01-01')

    args = parser.parse_args(sys.argv[1:])
    args = vars(args)
    # print(args)

    if args['coord']:
        lat, lon = args['coord']

    m = WRAPI_merra2()
    m.set_debug(True)

    if args['date'] and lat and lon:
        m.sam_date(lat, lon, args['date'])

    if args['month'] and lat and lon:
        m.sam_month(lat, lon, args['month'])

    if args['year'] and lat and lon:
        m.sam_year(lat, lon, args['year'])

    if args['auto']:
        # 中国经纬度范围：
        for j in range(1, 361+1):
            for i in range(1, 576+1):
                lat = -90.0 + (1.0/2.0)*(j-1)
                lon = -180 + (5.0/8.0)*(i-1)
                if 18.0 < lat < 44.0 and 73.66 < lon < 128.0:
                    if m.check_debug():
                        print("getting: 坐标: %s, %s" % (lat,lon))
                    m.sam_year(lat, lon, args['auto'])
                    #time.sleep(2)




