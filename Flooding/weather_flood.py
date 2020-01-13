import pandas as pd
import numpy as np

import datetime
from datetime import timedelta

import seaborn as sns
import matplotlib.pyplot as plt


class flooding:
    '''
    Performs different analysis related flooding
    '''

    def load_data(self):
        '''
        Loads data from the file
        '''
        xls = pd.ExcelFile("C:\\Users\\")
        self.df = pd.read_excel(xls, 'Raw Severe Data')

        # df1 = self.df['EVENT_TYPE'].unique()
        # print(np.sort(df1))

    def selected_data(self, interesting_county):
        '''
        Slices relevant data as needed
        '''

        self.df = self.df.loc[self.df['CORRECTED_NAME'].isin(interesting_county)]

    def find_event_seq(self, interesting_events, date_difference):
        '''
        Returns events preceding to any flooding events within certain days window
        '''

        f = 0
        p = 0
        df_casual = pd.DataFrame()
        df = self.df
        df = df.sort_values(by='BEGIN_DATE_TIME', ascending=True)
        df.index = range(len(df.index))
        for i in range(1, len(df)):
            if (df.loc[i, 'EVENT_TYPE']) in interesting_events:
                j = 1
                diff = df.loc[i, 'BEGIN_DATE_TIME'] - df.loc[i - j, 'BEGIN_DATE_TIME']

                while (diff.days < date_difference):
                    if ((df.loc[i, 'CORRECTED_NAME']) == (df.loc[i - j, 'CORRECTED_NAME'])) & (
                            df.loc[i - j, 'EVENT_TYPE'] not in interesting_events):
                        df_casual.loc[p, 'start_time_1st_event'] = df.loc[i - j, 'BEGIN_DATE_TIME']
                        df_casual.loc[p, 'start_time_2nd_event'] = df.loc[i, 'BEGIN_DATE_TIME']
                        df_casual.loc[p, '1st_event'] = df.loc[i - j, 'EVENT_TYPE']
                        df_casual.loc[p, '2nd_event'] = df.loc[i, 'EVENT_TYPE']
                        df_casual.loc[p, "CORRECTED_NAME"] = df.loc[i, 'CORRECTED_NAME']
                        f = f + 1

                        p = p + 1
                    j = j + 1
                    if (i - j < 0): break
                    diff = df.loc[i, 'BEGIN_DATE_TIME'] - df.loc[i - j, 'BEGIN_DATE_TIME']
            # if (f>100): break

        df_casual = df_casual.reset_index(drop=True)
        print(df_casual)
        df_casual.to_csv('C:\\Users\\mohammad.sharif\\Desktop\\DS_models\\Flooding\\causing_event.csv')
        return (df_casual)

    def find_flood_difference(self, interesting_events):
        '''
        Returns yearly flooding event count changes
        '''

        df = self.df
        df = df.loc[df['EVENT_TYPE'].isin(interesting_events)]

        # df['year'] = pd.DatetimeIndex(df['BEGIN_DATE_TIME']).year
        df['Month'] = pd.DatetimeIndex(df['BEGIN_DATE_TIME']).month
        df['Day'] = pd.DatetimeIndex(df['BEGIN_DATE_TIME']).day

        # df = df.groupby(['CZ_NAME','year']).size().reset_index(name='counts')
        df = df.groupby(['CORRECTED_NAME', 'Year']).size().reset_index(name='counts')
        # mean = df.groupby(['CZ_NAME','year']).agg(Mean=('counts', 'mean'))
        # mean = df.groupby(['CORRECTED_NAME','year']).agg(Mean=('counts', 'mean'))
        # std = df.groupby(['CORRECTED_NAME','year']).agg(Std=('counts', 'std'))

        df = df.sort_values(by=['CORRECTED_NAME', 'Year'], ascending=True)
        # print(df)

        demean = lambda x: (x - x.mean()) / np.std(x)
        counts_normalized = df.groupby(['CORRECTED_NAME']).transform(demean)["counts"]

        df['counts_normalized'] = counts_normalized
        print(df.loc[df["CORRECTED_NAME"] == 'RICHMOND'])
        return (df)

    def visualize_hitmap(self, df_vz):
        '''
        Draws hitmap visualization of yearly flooding changes by county and year
        '''

        df = df_vz

        # define the plot
        fig, ax = plt.subplots(figsize=(12, 7))

        # Add title to the Heat map
        title = "Flooding Changes(Consecutive Year)"

        # Set the font size and the distance of the title
        plt.title(title, fontsize=18)
        ttl = ax.title
        ttl.set_position([0.5, 1.05])

        # Hide ticks for X & Y axis
        ax.set_xticks([])
        ax.set_yticks([])

        df['counts_normalized_diff'] = df.counts_normalized.diff() + 3

        # Draw hitmap
        heatmap2_data = pd.pivot_table(df, values='counts_normalized_diff', index=['CORRECTED_NAME'], columns='Year')
        sns.heatmap(heatmap2_data, cmap="BuGn")
        plt.show()

        return

    def flooding_test(self, interesting_events):
        '''
        Tets Flooding situtation
        '''

        df = self.df
        df = df.loc[df['EVENT_TYPE'].isin(interesting_events)]
        df = df.groupby(['Year']).size().reset_index(name='counts')
        df = df.sort_values(by=['Year'], ascending=True)
        print(df)
        data = df.loc[:15, 'counts'].mean(), df.loc[15:, 'counts'].mean()
        data = df.loc[:10, 'counts'].mean(), df.loc[10:, 'counts'].mean()
        print(data)

    def plot_flood_vs_other(self):
        '''
        Draws scatter plots for different events with flooding for comparison
        '''

        df = self.df
        df = df[df['CORRECTED_NAME'] == 'STAFFORD']
        # distinct_county = df.CORRECTED_NAME.unique()
        distinct_year = df.Year.unique()
        distinct_EVENT_TYPE = df.EVENT_TYPE.unique()

        df = df.groupby(['CORRECTED_NAME', 'Year', 'EVENT_TYPE']).size().reset_index(name='counts')

        county = 'STAFFORD'
        tuple = []
        for year in distinct_year:
            for event in distinct_EVENT_TYPE:
                list = [year, event]
                tuple.append(list)

        grouped_set = []
        for index, row in df.iterrows():
            grouped_set.append([row['Year'], row['EVENT_TYPE']])
        # print(grouped_set)

        for element in tuple:
            if element not in grouped_set:
                df = df.append({'CORRECTED_NAME': county, 'Year': element[0], 'EVENT_TYPE': element[1], 'counts': 0},
                               ignore_index=True)

        year = np.asarray(sorted(df.Year.unique()))
        print(year)
        Dense_Fog = np.asarray(df[df['EVENT_TYPE'] == 'Dense Fog'].sort_values(['Year']).counts)
        print(Dense_Fog)
        Heavy_Rain = np.asarray(df[df['EVENT_TYPE'] == 'Heavy Rain'].sort_values(['Year']).counts)
        print(Heavy_Rain)
        Heavy_Snow = np.asarray(df[df['EVENT_TYPE'] == 'Heavy Snow'].sort_values(['Year']).counts)
        print(Heavy_Snow)
        High_Wind = np.asarray(df[df['EVENT_TYPE'] == 'High Wind'].sort_values(['Year']).counts)
        print(High_Wind)
        # Hurricane_Typhoon = np.asarray(df[df['EVENT_TYPE'] == 'Hurricane(Typhoon)'].sort_values(['Year']).counts)
        # print(Hurricane_Typhoon)
        Ice_Storm = np.asarray(df[df['EVENT_TYPE'] == 'Ice Storm'].sort_values(['Year']).counts)
        print(Ice_Storm)
        Winter_Storm = np.asarray(df[df['EVENT_TYPE'] == 'Winter Storm'].sort_values(['Year']).counts)
        print(Winter_Storm)
        Flash_Flood = np.asarray(df[df['EVENT_TYPE'] == 'Flash Flood'].sort_values(['Year']).counts)
        print(Flash_Flood)
        Flood = np.asarray(df[df['EVENT_TYPE'] == 'Flood'].sort_values(['Year']).counts)
        print(Flood)

        #['Flood', 'Flash Flood']

        df = pd.DataFrame({'Year': year, 'Dense_Fog': Dense_Fog, 'Heavy_Rain': Heavy_Rain, 'Heavy_Snow': Heavy_Snow,
                           'High_Wind': High_Wind, 'Ice Storm': Ice_Storm, 'Winter_Storm': Winter_Storm,
                           'Flash_Flood': Flash_Flood, 'Flood': Flood})

        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22",
                  "#17becf"]
        i = 0
        # fig, ax = plt.subplots(1, 1)
        fig, axs = plt.subplots(8, sharey=True, squeeze=False)
        # axs.figure.set_size_inches(5, 8)
        fig.suptitle('Yearly Event Counts')
        for column in df.drop('Year', axis=1):
            print(df['Year'])
            axs[i][0].scatter(df['Year'], df[column], color=colors[i], label=column)
            # axs[i][0].plot(df['Year'], df[column], color=colors[i], label=column)
            # axs[i][0].legend(loc='center left', bbox_to_anchor=(0.82, 1), fancybox=True, shadow=True, ncol=3)
            axs[i][0].legend(loc='left', bbox_to_anchor=(0.82, 1), fancybox=True, shadow=True)
            axs[i][0].figure.set_size_inches(15, 3)
            # axs[i][0].set_autoscalex_on(False)
            axs[i][0].autoscale(False, axis='X')
            axs[i][0].axis('tight')
            axs[i][0].set_ylim(bottom=1)
            if (i < 7):
                axs[i][0].set_xticklabels([])
            # axs[i][0].autoscale(enable=False, axis='both', tight=False)
            i = i + 1

        plt.show()




def main():
    interesting_events = ['Flood', 'Flash Flood']
    interesting_county = ['CAROLINE', 'ESSEX', 'FREDERICKSBURG', 'GLOUCESTER', 'KING GEORGE', 'KING AND QUEEN',
                          'KING WILLIAM', 'LANCASTER', 'MATHEWS', 'MIDDLESEX', 'NORTHUMBERLAND', 'SPOTSYLVANIA',
                          'STAFFORD', 'RICHMOND', 'WESTMORELAND']
    date_duration = 7
    flood = flooding()
    flood.load_data()
    # flood.selected_data(interesting_county)
    # event_seq_df = flood.find_event_seq(interesting_events, date_duration)
    # event_diff_df = flood.find_flood_difference(interesting_events)
    # flood.visualize_hitmap(event_diff_df)
    # flood.flooding_test(interesting_events)
    flood.plot_flood_vs_other()


if __name__ == '__main__':
    main()
