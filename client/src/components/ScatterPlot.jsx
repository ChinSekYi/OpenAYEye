// install (please try to align the version of installed @nivo packages)
// yarn add @nivo/scatterplot
import { ResponsiveScatterPlot } from '@nivo/scatterplot';
import { useTheme } from "@mui/material";
import { tokens } from "../theme";
import { mockScatterData } from "../data/mockData";

// make sure parent container have a defined height when using
// responsive component, otherwise height will be 0 and
// no chart will be rendered.
// website examples showcase many properties,
// you'll often use just a few of them.
const Scatter = ({data = mockScatterData, isDashboard = false}) => {
    const theme = useTheme();
	  const colors = tokens(theme.palette.mode);

    return <ResponsiveScatterPlot
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
        margin={{ top: 60, right: 140, bottom: 70, left: 90 }}
        xScale={{ type: 'linear', min: 'auto', max: 'auto' }}
        // xFormat=">-.2f"
        yScale={{ type: 'linear', min: 'auto', max: 'auto' }}
        yFormat=">-.2f"
        blendMode="multiply"
        nodeSize={12}
        axisTop={null}
        axisRight={null}
        axisBottom={{
            orient: 'bottom',
            tickSize: 5,
            tickPadding: 5,
            tickRotation: 0,
            legend: 'Variable',
            legendPosition: 'middle',
            legendOffset: 46,
            truncateTickAt: 0
        }}
        axisLeft={{
            orient: 'left',
            tickSize: 5,
            tickPadding: 5,
            tickRotation: 0,
            legend: 'Weight',
            legendPosition: 'middle',
            legendOffset: -60,
            truncateTickAt: 0
        }}
        legends={[
            {
                anchor: 'bottom-right',
                direction: 'column',
                justify: false,
                translateX: 130,
                translateY: 0,
                itemWidth: 100,
                itemHeight: 12,
                itemsSpacing: 5,
                itemDirection: 'left-to-right',
                symbolSize: 12,
                symbolShape: 'circle',
                effects: [
                    {
                        on: 'hover',
                        style: {
                            itemOpacity: 1,
                          },
                    }
                ]
            }
        ]}
    />
    }

export default Scatter;