// install (please try to align the version of installed @nivo packages)
// yarn add @nivo/heatmap
import { ResponsiveHeatMap } from '@nivo/heatmap';
import { useTheme } from "@mui/material";
import { tokens } from "../theme";
import { mockHeatMapData as data } from "../data/mockData";

// make sure parent container have a defined height when using
// responsive component, otherwise height will be 0 and
// no chart will be rendered.
// website examples showcase many properties,
// you'll often use just a few of them.
const HeatMap = ({isDashboard = false}) => {
    const theme = useTheme();
	const colors = tokens(theme.palette.mode);

    return <ResponsiveHeatMap
        data={data}
        theme={{
            axis: {
              domain: {
                line: {
                  stroke: colors.gray[100],
                },
              },
              legend: {
                text: {
                  fill: colors.gray[100],
                },
              },
              ticks: {
                line: {
                  stroke: colors.gray[100],
                  strokeWidth: 1,
                },
                text: {
                  fill: colors.gray[100],
                },
              },
            },
            legends: {
              text: {
                fill: colors.gray[100],
              },
            },
            tooltip: {
              container: {
                color: colors.primary[500],
              },
            },
          }}
        margin={{ top: 60, right: 90, bottom: 60, left: 90 }}
        valueFormat=">-.2s"
        axisTop={{
            tickSize: 5,
            tickPadding: 5,
            tickRotation: -90,
            legend: '',
            legendOffset: 46,
            truncateTickAt: 0
        }}
        axisRight={{
            tickSize: 5,
            tickPadding: 5,
            tickRotation: 0,
            legend: 'country',
            legendPosition: 'middle',
            legendOffset: 70,
            truncateTickAt: 0
        }}
        axisLeft={{
            tickSize: 5,
            tickPadding: 5,
            tickRotation: 0,
            legend: 'country',
            legendPosition: 'middle',
            legendOffset: -72,
            truncateTickAt: 0
        }}
        colors={{
            type: 'diverging',
            scheme: 'red_yellow_blue',
            divergeAt: 0.5,
            minValue: -100000,
            maxValue: 100000
        }}
        emptyColor="#555555"
        legends={[
            {
                anchor: 'bottom',
                translateX: 0,
                translateY: 30,
                length: 400,
                thickness: 8,
                direction: 'row',
                tickPosition: 'after',
                tickSize: 3,
                tickSpacing: 4,
                tickOverlap: false,
                tickFormat: '>-.2s',
                title: 'Value →',
                titleAlign: 'start',
                titleOffset: 4
            }
        ]}
    />
    }

export default HeatMap;