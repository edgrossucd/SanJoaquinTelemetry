# track.py 
# methods for dealing with a single acoustic telemetry track

import sys, os
import numpy as np
import pandas as pd
#from segment import Segment
from pylab import date2num as d2n
from datetime import datetime
import matplotlib.pyplot as plt
import stompy.plot.cmap as cmap
import pdb

cm_age = plt.get_cmap('spring')

dt_signal = 5

class Track(object):

    # currently not using defined types
    detection_dtype = [('ID',np.int32),
                       ('xyz',(np.float64,3)),
                       ('sec',np.float64),
                       ('dnum',np.float64),
                       ('node_count',np.int32)]

    def __init__(self, 
                 df_track = None,
                 grd = None):
        self.df_track = df_track
        self.grd = grd
        # automatically convert to defined type
        self.define_track() # can convert to defined type

    def define_track(self, set_depth=True):
        grd = self.grd
        self.ID = self.df_track['ID'].values[0]
        self.ndetects = len(self.df_track)
#       self.detects = np.zeros(ndetects, self.detection_dtype)
        # save redundant ID information for each detection
#       self.detects['ID'] = self.df_track['ID'].values
#       xyz = np.stack((self.df_track['X'].values, self.df_track['Y'].values,
#                       self.df_track['Z'].values),axis=-1)
#       self.detects['sec'] = self.df_track['Epoch_Sec'].values
#       self.detects['node_count'] = self.df_track['Node_count'].values
        dstrings = self.df_track['dt'].values
        dtimes = [datetime.strptime(ds, '%Y-%m-%d %H:%M:%S') for ds in dstrings]
#       self.detects['dnum'] = d2n(dtimes)
        dnums = d2n(dtimes)
        self.df_track = self.df_track.assign(dnums = dnums)
        if set_depth:
            depth = np.nan*np.ones(self.ndetects, np.float64)
            idetect = np.nan*np.ones(self.ndetects, np.int32)
            for nd in range(self.ndetects):
                xy = [self.df_track['X'].values[nd],
                      self.df_track['Y'].values[nd]]
                i = grd.select_cells_nearest(xy, inside=True)
                idetect[nd] = i
                if i >= 0:
                    depth[nd] = grd.cells['depth'][i]
            self.df_track = self.df_track.assign(depth = depth)
            self.df_track = self.df_track.assign(i = idetect)
        self.rec_track = self.df_track.to_records()
#       self.df_track = self.df_track.assign(xyz = xyz)

    # to do:
    # - pass in DEM
    # - pass in elevation time series
    # - set depth at xm,ym,tm = max(0.0,eta(tm) - bed_elev(xm,ym))
    # - use interpolated velocity to get uh, vh - hydro velocity 
    # - difference to get swimming speed

    def make_segments(self, set_depth=True):
        grd = self.grd
        self.nsegments = self.ndetects - 1
        if self.nsegments < 1:
            return
        x1 = np.nan*np.ones(self.nsegments, np.float64)
        y1 = np.nan*np.ones(self.nsegments, np.float64)
        x2 = np.nan*np.ones(self.nsegments, np.float64)
        y2 = np.nan*np.ones(self.nsegments, np.float64)
        xm = np.nan*np.ones(self.nsegments, np.float64)
        ym = np.nan*np.ones(self.nsegments, np.float64)
        t1 = np.nan*np.ones(self.nsegments, np.float64)
        t2 = np.nan*np.ones(self.nsegments, np.float64)
        tm = np.nan*np.ones(self.nsegments, np.float64)
        dt = np.nan*np.ones(self.nsegments, np.float64)
        dist = np.nan*np.ones(self.nsegments, np.float64)
        speed = np.nan*np.ones(self.nsegments, np.float64)
        u = np.nan*np.ones(self.nsegments, np.float64)
        v = np.nan*np.ones(self.nsegments, np.float64)
        idetect = np.nan*np.ones(self.nsegments, np.int32)
        for nd in range(1,self.ndetects):
            ns = nd-1
            x1[ns] = self.rec_track['X'][nd-1]
            y1[ns] = self.rec_track['Y'][nd-1]
            x2[ns] = self.rec_track['X'][nd]
            y2[ns] = self.rec_track['Y'][nd]
            t1[ns] = self.rec_track['Sec'][nd-1]
            t2[ns] = self.rec_track['Sec'][nd]
            dx = x2[ns] - x1[ns]
            dy = y2[ns] - y1[ns]
            xm[ns] = 0.5*(x1[ns]+x2[ns])
            ym[ns] = 0.5*(y1[ns]+y2[ns])
            tm[ns] = 0.5*(t1[ns]+t2[ns])
            dt[ns] = t2[ns] - t1[ns]
            dist[ns] = np.sqrt(dx*dx + dy*dy)
            if dt[ns] > 0.0:
                speed[ns] = dist[ns]/dt[ns] 
                u[ns] = dx/dt[ns]
                v[ns] = dy/dt[ns]
        self.df_seg = pd.DataFrame({'x1':x1, 'x2':x2, 'xm':xm,
                                    'y1':y1, 'y2':y2, 'ym':ym,
                                    't1':t1, 't2':t2, 'tm':tm,
                                    'dt':dt, 'dist':dist,
                                    'speed':speed, 'u':u, 'v':v})
        if set_depth:
            depth = np.nan*np.ones(self.nsegments, np.float64)
            for ns in range(self.nsegments):
                xy = [xm[ns], ym[ns]]
                i = grd.select_cells_nearest(xy, inside=True)
                idetect[ns] = i
                if i >= 0:
                    depth[ns] = grd.cells['depth'][i]
            self.df_seg = self.df_seg.assign(depth = depth)
            self.df_seg = self.df_seg.assign(i = idetect)

        self.rec_seg = self.df_seg.to_records()

    def nondetects(self):
        grd = self.grd
        xnd = []
        ynd = []
        ncells = len(grd.cells['depth'])
        non_detects_i_tr = np.zeros(ncells, np.int32)
        for nr, rseg in enumerate(self.rec_seg):
            seg = self.rec_seg[nr]
            dt = self.rec_seg['dt'][nr]
            if dt > dt_signal+1:
                t1 = self.rec_seg['t1'][nr]
                t2 = self.rec_seg['t2'][nr]
                #int_times = np.arange(t1, t2, dt_signal)
                nint = int(np.rint((t2-t1)/dt_signal)) - 1
                x1 = self.rec_seg['x1'][nr]
                x2 = self.rec_seg['x2'][nr]
                y1 = self.rec_seg['y1'][nr]
                y2 = self.rec_seg['y2'][nr]
                dx_nd = (x2 - x1)/float(nint+1)
                dy_nd = (y2 - y1)/float(nint+1)
                #print self.ID,nint,nr
                #print x1,x2,dx_nd
                #print y1,y2,dy_nd
                if nint < 120: # 10 minute cutoff for nondetect filling
                    xint = [x1 + n*dx_nd for n in range(1,nint)]
                    yint = [y1 + n*dy_nd for n in range(1,nint)]
                    xnd = xnd + xint
                    ynd = ynd + yint

        for nd in range(len(xnd)):
            xy = [xnd[nd], ynd[nd]]
            i = grd.select_cells_nearest(xy)
            if i >= 0:
                non_detects_i_tr[i] += 1

        return non_detects_i_tr

    def track_quality_metrics(self):
        grd = self.grd
        tr = self.rec_track
        seg = self.rec_seg
        detects = len(tr['X'])
        possible_detects = (tr['Sec'][-1] - tr['Sec'][0])/dt_signal
        non_detects = possible_detects - detects
        frac_detects = float(detects)/float(possible_detects)
        min_depth = np.min(tr['depth'])
        velocity_sd = self.velocity_sd()
        max_speed = np.max(np.absolute(seg['speed']))
        detects_outside_grid = self.detects_outside_grid()
        mnames = ['ID','non_detects','frac_detects','min_depth','vel_sd',
                  'invalid_cell']
        metrics = [self.ID, non_detects, frac_detects, min_depth, velocity_sd, 
                   detects_outside_grid]
        return mnames, metrics
        print self.ID, non_detects, frac_detects, min_depth, velocity_sd, \
              detects_outside_grid

    def detects_outside_grid(self):
        ii = self.rec_track['i']
        outside = sum(np.isnan(ii))

        return outside

    def velocity_sd(self):
        dspeed = np.zeros(self.ndetects-1, np.float64)
        for nr in range(self.nsegments-1):
            seg1 = self.rec_seg[nr]
            seg2 = self.rec_seg[nr+1]
            du  = seg1['u'] - seg2['u']
            dv  = seg1['v'] - seg2['v']
            dspeed[nr] = np.sqrt(du*du + dv*dv)

        return np.std(dspeed)

    def cell_detects(self):
        grd = self.grd
        ncells = len(grd.cells['depth'])
        detects_i_tr = np.zeros(ncells, np.int32)
        for nd in range(self.ndetects):
            tr = self.rec_track[nd]
            xy = [tr['X'],tr['Y']]
            i = grd.select_cells_nearest(xy)
            if i >= 0:
                detects_i_tr[i] += 1
           
        return detects_i_tr

    def cell_speed(self):
        grd = self.grd
        ncells = len(grd.cells['depth'])
        detects_i_tr = np.zeros(ncells, np.int32)
        speed_sum_i_tr = np.zeros(ncells, np.float64)
        for ns in range(self.nsegments):
            seg = self.rec_seg[ns]
            xy = [seg['xm'],seg['ym']]
            i = grd.select_cells_nearest(xy)
            if i >= 0:
                detects_i_tr[i] += 1
                speed_sum_i_tr[i] += seg['speed']
           
        return detects_i_tr, speed_sum_i_tr

    def cell_occupancy(self):
        grd = self.grd
        occupancy_i = []
        for nd in range(self.ndetects):
            tr = self.rec_track[nd]
            xy = [tr['X'],tr['Y']]
            i = grd.select_cells_nearest(xy)
            occupancy_i.append(i)

        occupancy_i = np.unique(occupancy_i)

        return occupancy_i

    def plot_speed(self, variable, ax, color_by=None, **kwargs):
        seg = self.rec_seg
        nseg = self.nsegments
        if color_by is None:
            self.quiv = ax.quiver(seg.xm, seg.ym, seg.u, seg.v, 
                                  units='x', scale=0.100, color='r',
                                  headlength=4.0, headwidth=3.0)
        else:
            if color_by == 'age':
                # offset time by 1 second
                time_from_entry = seg['tm'] - seg['tm'][0] + 1
                log_age = np.log10(time_from_entry)
                age_ticks = [10,100,1000,10000]
                ticks = [np.log10(a) for a in age_ticks]
                #tick_labels = [str(a) for a in age_ticks]
                tick_labels = ['$\mathregular{10^%d}$'%a for a in ticks]
                log_max_age = ticks[-1]
                color = log_age
                cmap_quiv = cm_age
                label = 'Time from Entry (seconds)'
            #else if color_by = 'quality':
            self.quiv = ax.quiver(seg.xm, seg.ym, seg.u, seg.v, color,
                                  units='x', scale=0.1, 
                                  cmap=cmap_quiv, #vmin=0., vmax=vmax, 
                                  headlength=4.0, headwidth=3.0)
            fig = plt.gcf()
            c1 = fig.colorbar(self.quiv)
            clims = [ticks[0],log_max_age]
            self.quiv.set_clim(clims)
            c1.set_label(label)
            c1.set_ticks(ticks)
            c1.set_ticklabels(tick_labels)

