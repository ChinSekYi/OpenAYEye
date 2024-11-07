import { Box } from "@mui/material";
import { Header, ScatterPlot} from "../../components";

const Scatter = () => {
  return (
    <Box m="20px">
      <Header title="Scatter Plot" subtitle="Simple Scatter Plot" />
      <Box height="75vh">
        <ScatterPlot />
      </Box>
    </Box>
  );
};

export default Scatter;
